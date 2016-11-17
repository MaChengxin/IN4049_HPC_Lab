for omega in 1.90 1.91 1.92 1.93 1.94 1.95 1.96 1.97 1.98 1.99
do
  make RELAX_FACTOR=omega run
  make clean
done
