# tilelang + tpu Guides

## Installation
After cloning this repo, we need to git apply a tvm patch like the following:

```bash
cp patches/tvm.patch 3rdparty/tvm/tvm.patch
cd 3rdparty/tvm
git apply tvm.patch
```

Then, we need to follow instruction of the upstream repo to build src code:

```bash
# we can run install_tpu.sh
./install_tpu.sh
```

After these commands, we should correctly `import tilelang` in python.

**Note** we need to recompile src code when modify the cpp code, like this

```bash
make -j 10 # you can specify the job count according to your machine status.
```
