cd build;
rm src/resources.o;
make -j12 && bin/game $@;

