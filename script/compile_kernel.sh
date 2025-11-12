echo Install EDQ CUDA Acceleration Engine.
cd kernels
pip uninstall edq_cuda_accel -y
python setup.py install
cd ..

echo Install AWQ Kernel for Test.
cd thirdparty/AWQ/kernels
python setup.py install
cd ../../..

echo Install Qserve Kernel for Test.
cd thirdparty/Qserve/qserve_kernels
python setup.py install
cd ../../..

# cd kernels; python setup.py install; cd ../
# cd thirdparty/AWQ/kernels/; python setup.py install; cd ../../..