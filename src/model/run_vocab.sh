SOURCEDIR=src
BUILDDIR=build
VOCAB_FILE=data/nyt/vocab/2016.txt
VOCAB_MIN_COUNT=1
VERBOSE=2
CORPUS=/projects/csl/viswanath/data/hgong6/DynamicEmbedding/NYT_year_slice/orig/nyt_2016.txt


gcc $SOURCEDIR/vocab_count.c -o $BUILDDIR/vocab_count -lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE