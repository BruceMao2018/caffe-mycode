if we can not find the library, we may need to add 'export LD_LIBRARY_PATH=$(pwd)' before g++
it will write the library into /etc/ld.so.conf',  you need to run 'ldconfig' after that. 
of course, you can add so in the path which defined in /etc/ld.so.conf
