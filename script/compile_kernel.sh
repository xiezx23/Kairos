cd kernels
# pip uninstall kairos_cuda_accel -y
python setup.py install
python lut_setup.py install
cd ..

cd thirdparty/AWQ/kernels
python setup.py install
cd ../../..

cd thirdparty/Qserve/qserve_kernels
python setup.py install
cd ../../..

cd thirdparty/marlin/
python setup.py install
cd ../../

cd thirdparty/QQQ/kernels
python setup.py install
cd ../../..
