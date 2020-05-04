CORPUS=data/nyt/nyt_2016.txt
COOCCURRENCE_FILE=data/nyt/cooccur.bin
COND=27
SCALE=1.59737663082
SOURCEDIR=src
BUILDDIR=build
VOCAB_FILE=data/nyt/vocab.txt
VOCAB_MIN_COUNT=200
VERBOSE=2
WINDOW_SIZE=5
BINARY=2
MEMORY=4.0

# count the vocabulary
#echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
#$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

# count the co-occurrences
gcc $SOURCEDIR/cooccur.c -o $BUILDDIR/cooccur -lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -distance-weighting 0 -verbose $VERBOSE -window-size $WINDOW_SIZE -cond $COND -scale $SCALE < $CORPUS >> $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -cooccur-file $COOCCURRENCE_FILE -distance-weighting 0 -verbose $VERBOSE -window-size $WINDOW_SIZE -cond $COND -scale $SCALE < $CORPUS


