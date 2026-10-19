"""Small, explicit adaptations of the supplied PPL SDK examples.

The emitted .pl file and original source hashes are retained for every trial.
No generated TileLang C is used as a PPL baseline.
"""
from pathlib import Path
import re

SOURCES = {
    "matmul": "examples/cxx/matmul/mm2_fp16.pl",
    "rmsnorm": "examples/cxx/llm/rmsnorm.pl",
    "rmsnorm-splitk": "examples/cxx/llm/rmsnorm.pl",
    "swiglu": "examples/cxx/llm/swi_glu.pl",
    "rope": "examples/cxx/llm/rope.pl",
    "flashattn": "examples/cxx/llm/flash_attention.pl",
    "llama-mlp": "examples/cxx/llm/mlp_multicore.pl",
    "paged-attention": "examples/cxx/llm/paged_attention_multicore.pl",
    "gptq-w4a16": "examples/cxx/matmul/w4a16_matmul_dq2.pl",
    "pipeline-matmul": "examples/cxx/matmul/mm2_fp16.pl",
    "pipeline-add": "samples/add_pipeline/ppl/add_pipeline.pl",
}


def render(root, case, config):
    op, shape, dtype = case["op"], case["shape"], case["dtype"]
    ctype = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}[dtype]
    pipeline = "enable_pipeline();" if config.get("stages", 1) == 2 else ""
    if op == "pipeline-add":
        width = shape[1]
        bw = config["block_width"]
        return f'''#include "ppl.h"
using namespace ppl;
__KERNEL__ void bench({ctype} *x,{ctype} *out) {{
 dim4 gs={{1,1,1,{width}}};dim4 bs={{1,1,1,{bw}}};
 auto gx=gtensor<{ctype}>(gs,GLOBAL,x);auto go=gtensor<{ctype}>(gs,GLOBAL,out);
 for(int w=0;w<{width};w+={bw}) {{
  {pipeline}
  auto a=tensor<{ctype}>(bs);auto b=tensor<{ctype}>(bs);dim4 off={{0,0,0,w}};
  dma::load(a,gx.sub_view(bs,off));tiu::fadd(b,a,0.25f);dma::store(go.sub_view(bs,off),b);
 }}
}}\n'''
    if op.startswith("elementwise-"):
        rows, width = shape
        br, bw = config["block_rows"], config["block_width"]
        primitive = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"}[op.split("-")[-1]]
        return f'''#include "ppl.h"
using namespace ppl;
__KERNEL__ void bench({ctype} *a, {ctype} *b, {ctype} *out) {{
  dim4 gs={{1,{rows},1,{width}}};
  auto ga=gtensor<{ctype}>(gs,GLOBAL,a);
  auto gb=gtensor<{ctype}>(gs,GLOBAL,b);
  auto go=gtensor<{ctype}>(gs,GLOBAL,out);
  dim4 bs={{1,{br},1,{bw}}};
  for(int r=0;r<{rows};r+={br}) {{
    for(int w=0;w<{width};w+={bw}) {{
      {pipeline}
      auto x=tensor<{ctype}>(bs); auto y=tensor<{ctype}>(bs); auto z=tensor<{ctype}>(bs);
      dim4 off={{0,r,0,w}};
      dma::load(x,ga.sub_view(bs,off)); dma::load(y,gb.sub_view(bs,off));
      tiu::{primitive}(z,x,y{',3' if primitive == 'fdiv' else ''});
      dma::store(go.sub_view(bs,off),z);
    }}
  }}
}}
'''
    source = (Path(root) / SOURCES[op]).read_text().split("__TEST__")[0]
    source = source.replace("__KERNEL__", "")
    if config.get("stages", 1) == 1:
        # enable_pipeline also supplies a firmware counter-retirement path.
        # Serial variants must go through build_runner's bounded command
        # transform/audit before cmodel validation and any PCIe admission.
        source = re.sub(r"(?:ppl::)?enable_pipeline\(\);", "", source)
    if op in ("matmul", "pipeline-matmul"):
        m, n, k = shape
        source = source.replace("fp16", ctype)
        source = source.replace(
            f"tensor<{ctype}> sub_left, sub_right;",
            f"auto sub_left=make_tensor<{ctype}>(left_max_shape,left_real_shape); "
            f"auto sub_right=make_tensor<{ctype}>(right_max_shape,right_real_shape);")
        # The vendor example's optional bias is outside the demo contract.
        source = re.sub(r"      dma::load_broadcast\(sub_bias,.*?;", "", source, flags=re.S)
        source = re.sub(
            r"tiu::fmm2_nn\(sub_res, sub_left, sub_right, sub_bias, true, DT_FP32,\s*do_relu, saturate, requant\);",
            "tiu::fmm2_nn(sub_res, sub_left, sub_right, true);", source)
        for axis in ("m", "n", "k"):
            source = re.sub(rf"const int block_{axis} = \d+;",
                            f"const int block_{axis} = {config['block_'+axis]};", source)
        source += f'''\n__KERNEL__ void bench({ctype} *a,{ctype} *b,{ctype} *out) {{
 mm2_{ctype}(out,a,b,b,{m},{k},{n});
}}\n'''
        return source
    if op in ("rmsnorm", "rmsnorm-splitk"):
        rows, width = shape
        source = source.replace("int block_c = LANE_NUM;", f"int block_c = {config['block_rows']};")
        source += f'''\n__KERNEL__ void bench({ctype} *x,{ctype} *weight,{ctype} *out) {{
 rms_norm_{ctype}(out,x,weight,1e-5f,true,1,{rows},1,{width},{config['block_width']});
}}\n'''
        return source
    if op == "swiglu":
        rows, width = shape
        source = re.sub(r"#ifdef __mars3__.*?#endif", f"#define DTYPE {ctype}", source, flags=re.S)
        source = source.replace("int block_w = 16;", f"int block_w = {config['block_width']};")
        source = source.replace("int block_c = 64;", f"int block_c = {config['block_rows']};")
        source += f'''\n__KERNEL__ void bench({ctype} *gate,{ctype} *up,{ctype} *out) {{
 swi_glu_kernel(out,gate,up,{rows},{width});
}}\n'''
        return source
    if op == "rope":
        rows, width = shape
        source = source.replace("fp32", ctype)
        # SDK declares half-width global tables but accesses width elements.
        # Contract uses full-width tables with each pair's angle repeated.
        source = source.replace("W / 2};", "W};")
        source = source.replace("int block_w = 16;", f"int block_w = {config['block_width']};")
        source = source.replace("int block_c = 64;", f"int block_c = {config['block_rows']};")
        source += f'''\n__KERNEL__ void bench({ctype} *x,{ctype} *cos,{ctype} *sin,{ctype} *out) {{
 rope_kernel(out,x,cos,sin,{rows},{width});
}}\n'''
        return source
    if op in ("flashattn", "paged-attention"):
        if op == "flashattn":
            b, seq, heads, d = shape
        else:
            blocks, qh, kvh, d = shape
            b, seq, heads = qh, blocks * 16, 1
            source = (Path(root) / "examples/cxx/llm/flash_attention_GQA_multicore.pl"
                     ).read_text().split("__TEST__")[0].replace("__KERNEL__", "")
            source = source.replace("ppl::set_block_num_max();", "ppl::set_block_num(1);")
            if config.get("stages", 1) == 1:
                source = re.sub(r"(?:ppl::)?enable_pipeline\(\);", "", source)
        # Keep the vendor online-softmax algorithm and BSHD layout.
        source = source.replace("const int block_m = 32;",
                                f"const int block_m = {config['block_m']};")
        source = source.replace("const int block_k = 64;",
                                f"const int block_k = {config['block_n']};")
        if op == "paged-attention":
            source = source.replace("const int block_m = 64;", "const int block_m = 1;")
            source = source.replace("const int block_k = 256;",
                                    f"const int block_k = {config['block_n']};")
            source = source.replace("const int block_h = 8;", "const int block_h = 1;")
        source = source.replace("const int dmax = 128;", f"const int dmax = {d};")
        source = source.replace("const int block_h = 32;", "const int block_h = 1;")
        source = source.replace("const int block_h = 16;", "const int block_h = 1;")
        if op == "flashattn":
            # The SDK sample allocates K/V with q_shape (block_m rows).
            # Larger KV tiles then exceed that allocation once they span
            # multiple 64-lane rows. Preserve its algorithm with correct
            # max-shape bounds for the actually requested KV tile.
            source = source.replace(
                "dim4 q_shape = {block_h, block_m, 1, dmax};",
                "dim4 q_shape = {block_h, block_m, 1, dmax};\n"
                "  dim4 kv_shape = {block_h, block_k, 1, dmax};")
            source = source.replace("make_tensor<T>(q_shape, kvi_real_local_shape)",
                                    "make_tensor<T>(kv_shape, kvi_real_local_shape)")
            source += f'''\n__KERNEL__ void bench({ctype} *q,{ctype} *k,{ctype} *v,{ctype} *mask,{ctype} *out) {{
 flash_attention<{ctype}>(out,q,k,v,mask,{b},{seq},{seq},{d},{heads},{d**-0.5}f,1);
}}\n'''
        else:
            # Read-only paired operator: the supplied vendor paged kernel also
            # updates KV and optionally applies RoPE, absent in the demo.
            # Reuse SDK gather_h and its online attention loop, per head.
            rows = blocks * kvh * 16
            staged = qh * seq
            source += f'''\n__KERNEL__ void bench({ctype} *q,{ctype} *kc,{ctype} *vc,uint32 *ki,uint32 *vi,{ctype} *ks,{ctype} *vs,{ctype} *out) {{
 dim4 cs={{1,1,{rows},{d}}}; dim4 ss={{1,1,{staged},{d}}};
 dim4 is={{1,1,{staged},1}}; dim4 hs={{1,1,{seq},{d}}}; dim4 hi={{1,1,{seq},1}};
 auto gkc=gtensor<{ctype}>(cs,GLOBAL,kc);auto gvc=gtensor<{ctype}>(cs,GLOBAL,vc);
 auto gks=gtensor<{ctype}>(ss,GLOBAL,ks);auto gvs=gtensor<{ctype}>(ss,GLOBAL,vs);
 auto gki=gtensor<uint32>(is,GLOBAL,ki);auto gvi=gtensor<uint32>(is,GLOBAL,vi);
 for(int h=0;h<{qh};h++) {{
  dim4 off={{0,0,h*{seq},0}};
  dma::gather_h(gks.sub_view(hs,off),gkc,gki.sub_view(hi,off),0);
  dma::gather_h(gvs.sub_view(hs,off),gvc,gvi.sub_view(hi,off),0);
 }}
 for(int h=0;h<{qh};h++) {{
  flash_attention_gqa<{ctype}>(out+h*{d},q+h*{d},ks+h*{seq*d},vs+h*{seq*d},q,1,1,{seq},{d},1,1,{d**-0.5}f,0);
 }}
}}\n'''
        return source
    if op == "gptq-w4a16":
        m, n, k = shape
        source += f'''\n__KERNEL__ void bench(fp16 *x,uint8 *packed,uint32 *scale,fp16 *out) {{
 matmul_w4a16_rtrans_inner(out,x,packed,x,scale,false,{m},{k},{n},128,{min(m,32)},64,true,true);
}}\n'''
        return source
    if op == "llama-mlp":
        m, h, i = shape
        source = source.replace("fp16",
                                ctype).replace("DT_FP16",
                                               "DT_BF16" if dtype == "bfloat16" else "DT_FP16")
        source = source.replace("dma::fill(out_f32_local, 0);", "tiu::zero(out_f32_local);")
        source = source.replace("int b_loop = div_up(batch, LANE_NUM);",
                                f"int b_loop = div_up(batch, {config['block_rows']});")
        source = source.replace("b_idx * LANE_NUM", f"b_idx * {config['block_rows']}")
        source = source.replace("min(LANE_NUM, batch - b_offset)",
                                f"min({config['block_rows']}, batch - b_offset)")
        # Specialize the vendor multi-core reduction to one core. Its final
        # L2 all-reduce is an identity here; keep the same resident-input,
        # streaming-intermediate algorithm and store the final tile directly.
        source = source.split("int psum_op")[
            0] + f'''auto output_global=gtensor<{ctype}>(g_input_shape,GLOBAL,ptr_output);
 dma::store(output_global.sub_view(input_shape,g_offset),input_local);
 }} }} }}\n'''
        source += f'''\n__KERNEL__ void bench({ctype} *x,{ctype} *up,{ctype} *gate,{ctype} *down,{ctype} *out) {{
 llma_mlp_mulit_core_kernel_bf16(x,up,gate,down,out,{m},{h},{i},1,{config['block_rows']},{h},{config['block_i']});
}}\n'''
        return source
    raise NotImplementedError(f"baseline adapter missing for {op}")
