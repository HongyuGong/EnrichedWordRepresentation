// Enriched Word Embedding Training 
// Temporal Embedding


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    int word3;
    real val;
} CREC;

int write_header=1; //0=no, 1=yes; writes vocab_size/vector_size as first line for use with some libraries, such as gensim.
int verbose = 2; // 0, 1, or 2
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int condition_size = 40; // condition_size: z-dimension in tensor
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 0; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
int checkpoint_every = 0; // checkpoint the model for every checkpoint_every iterations. Do nothing if checkpoint_every <= 0
real eta = 0.02; // Initial learning rate: 0.05
//real lambda = 0.1; // weight on bias l2 regularizer
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *T, *Q, *gradsqW, *gradsqQ, *gradsqT, *cost;
long long num_lines, *lines_per_thread, vocab_size; 
char *vocab_file, *input_file, *save_W_file, *save_W_T_file, *save_W_cxt_file, *save_W_T_cxt_file, *save_gradsq_file;
real reg = 0.2; // 0.2 weight on (word, time) vector l2 regularizer
real year_reg = 0.2; // 1.0, weight on time vector l2 regularizer
real bias_reg = 0.0; // 0.0, weight on word-time BIAS l2 regularizer
real init_scale = 4.0; // best: 4.0 
real update_scale = 1.0;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
    long long a, b;

    /* Allocate space for word vectors and context word vectors, and correspodning gradsq */
 	// basic vectors for words and context words
    a = posix_memalign((void **)&W, 128, (2 * vocab_size * (vector_size + condition_size)) * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    // time vectors for words and context words
    a = posix_memalign((void **)&T, 128, (2 * condition_size * vector_size) * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for T\n");
        exit(1);
    }
    // (word, time) vectors for words and context words
    a = posix_memalign((void **)&Q, 128, (2 * condition_size * vocab_size * vector_size) * sizeof(real)); // Might perform better than malloc
	if (Q == NULL) {
        fprintf(stderr, "Error allocating memory for Q\n");
        exit(1);
    }
    // gradient square for W
    a = posix_memalign((void **)&gradsqW, 128, (2 * vocab_size * (vector_size + condition_size)) * sizeof(real)); // Might perform better than malloc
	if (gradsqW == NULL) {
        fprintf(stderr, "Error allocating memory for gradsqW\n");
        exit(1);
    }
    // gradient square for T
    a = posix_memalign((void **)&gradsqT, 128, (2 * condition_size * vector_size) * sizeof(real)); // Might perform better than malloc
	if (gradsqT == NULL) {
        fprintf(stderr, "Error allocating memory for gradsqT\n");
        exit(1);
    }
    // gradient square for Q
    a = posix_memalign((void **)&gradsqQ, 128, (2 * condition_size * vocab_size * vector_size) * sizeof(real)); // Might perform better than malloc
	if (gradsqQ == NULL) {
        fprintf(stderr, "Error allocating memory for gradsqQ\n");
        exit(1);
    }

    // initialization of W, T and Q
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * (vector_size + condition_size) + b] = init_scale * (rand() / (real)RAND_MAX - 0.5) / vector_size;
    for (b = vector_size; b < vector_size + condition_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * (vector_size + condition_size) + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * condition_size; a++) T[a * vector_size + b] = init_scale * (rand() / (real)RAND_MAX - 0.5) / vector_size;
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size * condition_size; a++) Q[a * vector_size + b] = init_scale * (rand() / (real)RAND_MAX - 0.5) / vector_size;

    // initialization of gradsqW, gradsqT, gradsqQ
    for (b = 0; b < vector_size + condition_size; b++) for (a = 0; a < 2 * vocab_size; a++) gradsqW[a * (vector_size + condition_size) + b] = 1.0;
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * condition_size; a++) gradsqT[a * vector_size + b] = 1.0;
    for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size * condition_size; a++) gradsqQ[a * vector_size + b] = 1.0;

}

inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

/* Train the GloVe model */
void *ewe_thread(void *vid) {
    long long a, b ,l1, l2, l3, l4, l5, l6, l7, l8;
    int neb;
    long long id = *(long long*)vid;
    CREC cr;
    real diff, fdiff, bias_fdiff_1, bias_fdiff_2, Wtemp1, Wtemp2, Ttemp1, Ttemp2, Qtemp1, Qtemp2;

    // file stores object cr
    FILE *fin;
    fin = fopen(input_file, "rb");
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;
    
    // word basis vector W
    real* W_updates1 = (real*)malloc((vector_size+condition_size) * sizeof(real));
    real* W_updates2 = (real*)malloc((vector_size+condition_size) * sizeof(real));
    // time vector T
    real* T_updates1 = (real*)malloc(vector_size * sizeof(real));
    // vector Q: |Q(l5 + b) - Q(l5 - vocab_size * vector_size + b)|
    real * Q_updates1 = (real *)malloc(vector_size * sizeof(real)); // 2: current + prev
    real * Q_updates2 = (real *)malloc(vector_size * sizeof(real)); // 2: current + prev


    for (a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        if (cr.word1 < 1 || cr.word2 < 1 || cr.word3 < 1) { continue; }

        /* location of words in W, gradsq & Q */
        // word 1 in W
        l1 = (cr.word1 - 1LL) * (vector_size + condition_size);
        // word 2 in W
        l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + condition_size);
        // time 1 in T
        l3 = (cr.word3 - 1LL) * vector_size;
        // word 1 at time in Q
        l5 = (cr.word3 - 1LL) * vocab_size * vector_size + (cr.word1 - 1LL) * vector_size;
        // word 2 at time in Q 
        l6 = condition_size * vocab_size * vector_size + (cr.word3 - 1LL) * vocab_size * vector_size + (cr.word2 - 1LL) * vector_size;
        // bias of word 1 in W
        l7 = vector_size + l1 + (cr.word3 - 1LL);
        // bias of word 2 in W
        l8 = vector_size + l2 + (cr.word3 - 1LL);

        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (b = 0; b < vector_size; b++) {
                diff += (W[b + l1] * T[b + l3] + Q[l5 + b]) * (W[b + l2] * T[b + l3] + Q[l6 + b]);
        	if (isnan(diff)) fprintf(stderr, "b: %d, diff: %f\n", b, diff);	
        }

        // dynamic: word-year bias
        diff += W[vector_size + l1 + (cr.word3-1LL)] + W[vector_size + l2 + (cr.word3-1LL)] - log(cr.val + 1); // add separate bias for each word
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff

        // Check for NaN and inf() in the diffs.
        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)) {
                fprintf(stderr, "word 1: %d\n", cr.word1);
                fprintf(stderr, "word 2: %d\n", cr.word2);
                fprintf(stderr, "word 3: %d\n", cr.word3);
        	fprintf(stderr, "diff is %f\n", diff);
	    	fprintf(stderr, "fdiff is %f\n", fdiff);
	    	fprintf(stderr, "value is %f\n", cr.val);
        	fprintf(stderr,"Caught NaN in diff for kdiff for thread. Skipping update");
        	//continue;
	    	exit(1);
        }

        cost[id] += fdiff * diff; // weighted squared error
        
        /* Adaptive gradient updates */
        fdiff *= eta; // for ease in calculating gradient
        real W_updates1_sum = 0;
        real W_updates2_sum = 0;
        real T_updates1_sum = 0;
        real Q_updates1_sum = 0;
        real Q_updates2_sum = 0;

        for (b = 0; b < vector_size; b++) {
            // W[l1+b]
            Wtemp1 = fdiff * (T[l3 + b] * (W[l2 + b] * T[l3 + b] + Q[l6 + b]));
            // W[l2+b]
            Wtemp2 = fdiff * (T[l3 + b] * (W[l1 + b] * T[l3 + b] + Q[l5 + b]));
            // T[l3+b]
            Ttemp1 = fdiff * (2 * (W[l1 + b] * W[l2 + b] * T[l3 + b]) + W[l1 + b] * Q[l6 + b] + W[l2 + b] * Q[l5 + b]);
            for (neb = 0; neb < condition_size; neb++) {
            	if (neb + 1 != cr.word3) {
            		l4 = neb * vector_size;
            		Ttemp1 += eta * year_reg * (T[l3 + b] - T[l4 + b]);
            	}
            }

            // Q[l5+b]
            Qtemp1 = fdiff * (W[l2 + b] * T[l3 + b] + Q[l6 + b]) + eta * reg * Q[l5 + b];
            // Q[l6+b]
            Qtemp2 = fdiff * (W[l1 + b] * T[l3 + b] + Q[l5 + b]) + eta * reg * Q[l6 + b];
            // ADAPTIVE UPDATE PARAMETERS
            W_updates1[b] = Wtemp1 / sqrt(gradsqW[b + l1]);
            W_updates2[b] = Wtemp2 / sqrt(gradsqW[b + l2]);
            T_updates1[b] = Ttemp1 / sqrt(gradsqT[b + l3]);
            //T_updates2[b] = Ttemp2 / sqrt(gradsqT[b + l4]);
            Q_updates1[b] = Qtemp1 / sqrt(gradsqQ[b + l5]);
            Q_updates2[b] = Qtemp2 / sqrt(gradsqQ[b + l6]);
            // sum of updates
            W_updates1_sum += W_updates1[b];
            W_updates2_sum += W_updates2[b];
            T_updates1_sum += T_updates1[b];
            Q_updates1_sum += Q_updates1[b];
            Q_updates2_sum += Q_updates2[b];

            // squared updates
            gradsqW[b + l1] += Wtemp1 * Wtemp1;
            gradsqW[b + l2] += Wtemp2 * Wtemp2;
            gradsqT[b + l3] += Ttemp1 * Ttemp1;
            gradsqQ[b + l5] += Qtemp1 * Qtemp1;
            gradsqQ[b + l6] += Qtemp2 * Qtemp2;         
        }

        if (!isnan(W_updates1_sum) && !isinf(W_updates1_sum) && \
        	!isnan(W_updates2_sum) && !isinf(W_updates2_sum) && \
        	!isnan(T_updates1_sum) && !isinf(T_updates1_sum) && \
        	!isnan(Q_updates1_sum) && !isinf(Q_updates1_sum) && \
        	!isnan(Q_updates2_sum) && !isinf(Q_updates2_sum)) {
            // UPDATE PARAMETERS
            for (b = 0; b < vector_size; b++) {
                W[b + l1] -= update_scale * W_updates1[b];
                W[b + l2] -= update_scale * W_updates2[b];
                T[b + l3] -= update_scale * T_updates1[b];
                Q[b + l5] -= update_scale * Q_updates1[b];
                Q[b + l6] -= update_scale * Q_updates2[b];
            }
        }
        
        // updates for bias terms in dynamic vectors
        if (check_nan(fdiff / sqrt(gradsqW[vector_size + l1 + (cr.word3 - 1LL)])) == 0.0) {
            fprintf(stderr, "gradsqW:", gradsqW[vector_size + l1 + (cr.word3 - 1LL)]);
        }
        bias_fdiff_1 = fdiff + eta * bias_reg * W[l7];
        W[l7] -= check_nan(update_scale * bias_fdiff_1 / sqrt(gradsqW[l7]));
        bias_fdiff_2 = fdiff + eta * bias_reg * W[l8];
        W[l8] -= check_nan(update_scale * bias_fdiff_2 / sqrt(gradsqW[l8]));
        
        bias_fdiff_1 *= bias_fdiff_1;
        gradsqW[l7] += bias_fdiff_1;
        bias_fdiff_2 *= bias_fdiff_2;
        gradsqW[l8] += bias_fdiff_2;
        fdiff *= fdiff;
    }
    free(W_updates1);
    free(W_updates2);
    free(T_updates1);
    free(Q_updates1);
    free(Q_updates2);
    
    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params(int nb_iter) {
    long long a, b, c;
    char format[20], format2[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH], output_file2[MAX_STRING_LENGTH];
    char output_cxt_file[MAX_STRING_LENGTH], output_cxt_file2[MAX_STRING_LENGTH];
    char *word = malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    FILE *fid, *fid2, *fout, *fout2, *fout_cxt, *fout2_cxt, *fgs;
    
    if (use_binary > 0) { // Save parameters in binary file
        if (nb_iter <= 0) {
            sprintf(output_file,"%s.bin",save_W_file);
            sprintf(output_file2,"%s.bin",save_W_T_file);
        }
        else {
            sprintf(output_file,"%s.%03d.bin",save_W_file,nb_iter);
            sprintf(output_file2,"%s.%03d.bin",save_W_T_file,nb_iter);       
        }

        fout = fopen(output_file,"wb");
        fout2 = fopen(output_file2, "wb");
        if (fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        // write W
        for (a = 0; a < 2 * (long long) vocab_size * (vector_size + condition_size); a++) fwrite(&W[a], sizeof(real), 1, fout);
        // write T
        for (a = 0; a < 2 * (long long) condition_size * vector_size; a++) fwrite(&T[a], sizeof(real), 1, fout);
        // write Q
        for (a = 0; a < 2 * (long long) condition_size * vocab_size * vector_size; a++) fwrite(&Q[a], sizeof(real), 1, fout2);
        fclose(fout);
        fclose(fout2);
    }
    if (use_binary != 1) { // Save parameters in text file
        if (nb_iter <= 0) {
            sprintf(output_file,"%s.txt",save_W_file);
            sprintf(output_file2,"%s.txt",save_W_T_file);
            sprintf(output_cxt_file,"%s.txt",save_W_cxt_file);
            sprintf(output_cxt_file2,"%s.txt",save_W_T_cxt_file);
        }
        else {
            sprintf(output_file, "%s.%03d.txt", save_W_file, nb_iter);
            sprintf(output_file2, "%s.%03d.txt", save_W_T_file, nb_iter); 
            sprintf(output_cxt_file,"%s.%03d.txt",save_W_cxt_file,nb_iter);
            sprintf(output_cxt_file2,"%s.%03d.txt",save_W_T_cxt_file,nb_iter); 
        }

        fout = fopen(output_file,"wb");
        fout_cxt = fopen(output_cxt_file, "wb");
        fout2_cxt = fopen(output_cxt_file2, "wb");

        if (fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if (fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
	    if (write_header) {
            fprintf(fout, "%ld %d\n", vocab_size, vector_size);
            fprintf(fout_cxt, "%ld %d\n", vocab_size, vector_size);
        }
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid,format,word) == 0) return 1;
            // input vocab cannot contain special <unk> keyword
            if (strcmp(word, "<unk>") == 0) return 1;

            fprintf(fout, "%s",word);
            fprintf(fout_cxt, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout," %lf", W[a * (vector_size + condition_size) + b]);
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + condition_size) + b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + condition_size) + b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + condition_size) + b] + W[(vocab_size + a) * (vector_size + condition_size) + b]);
            if (model == 3) {// Save vector and context vectors in different files
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + condition_size) + b]);
                for (b = 0; b < vector_size; b++) fprintf(fout_cxt," %lf", W[(vocab_size + a) * (vector_size + condition_size) + b]);
            }
            fprintf(fout,"\n");
            fprintf(fout_cxt,"\n");

            if (fscanf(fid,format,word) == 0) return 1;
        }
        // save condition vectors
        for (c=0; c < condition_size; c++) {
            fprintf(fout, "%d",c);
            fprintf(fout_cxt, "%d",c);
            if (model == 0) { // save all including bias
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", T[c * vector_size + b]);
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", T[(c + condition_size) * vector_size + b]);
            }
        	if (model == 1) {
        		for (b = 0; b < vector_size; b++) fprintf(fout," %lf", T[c * vector_size + b]);
        	}
            if (model == 2) { // not saving bias
                // same time vectors
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", T[c * vector_size + b] + T[(c + condition_size) * vector_size + b]);
            }
            if (model == 3) { // save vector and context vectors into different files
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", T[c * vector_size + b]);
                for (b = 0; b < vector_size; b++) fprintf(fout_cxt," %lf", T[(c + condition_size) * vector_size + b]);  
            }
            fprintf(fout,"\n");
            fprintf(fout_cxt,"\n");
        }
        // save word-time vectors
        fid2 = fopen(vocab_file, "r");
        sprintf(format2,"%%%ds",MAX_STRING_LENGTH);
        fout2 = fopen(output_file2,"wb");
        if (fout2 == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_T_file); return 1;}
        for (a = 0; a < vocab_size; a++) {
        	if (fscanf(fid2,format2,word) == 0) return 1;
        	if (strcmp(word, "<unk>") == 0) return 1;
        	for (c=0; c < condition_size; c++) {
        		fprintf(fout2, "%s-%d", word, c);
                fprintf(fout2_cxt, "%s-%d", word, c);
        		if (model == 0) {
        			for (b=0; b<vector_size; b++) fprintf(fout2, " %lf", Q[c * vocab_size * vector_size + a * vector_size + b]);
        			for (b=0; b<vector_size; b++) fprintf(fout2, " %lf", Q[(c+condition_size) * vocab_size * vector_size + a * vector_size + b]);
        		}
        		if (model == 1) {
        			for (b=0; b<vector_size; b++) fprintf(fout2, " %lf", Q[c * vocab_size * vector_size + a * vector_size + b]);
        		}
        		if (model == 2) {
        			for (b=0; b<vector_size; b++) fprintf(fout2, " %lf", Q[c * vocab_size * vector_size + a * vector_size + b] +\
        			 Q[(c+condition_size) * vocab_size * vector_size + a * vector_size + b]);
        		}
                if (model == 3) {
                    for (b=0; b<vector_size; b++) fprintf(fout2, " %lf", Q[c * vocab_size * vector_size + a * vector_size + b]);
                    for (b=0; b<vector_size; b++) fprintf(fout2_cxt, " %lf", Q[(c+condition_size) * vocab_size * vector_size + a * vector_size + b]);
                }
                fprintf(fout2,"\n");
                fprintf(fout2_cxt, "\n");
        	}
        	if (fscanf(fid2,format2,word) == 0) return 1; // Eat irrelevant frequency entry
        }


        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + condition_size), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + condition_size), sizeof(real));
            word = "<unk>";

            int num_rare_words = vocab_size < 100 ? vocab_size : 100;

            // take average of least frequent words as the vector for the unknown word
            for (a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for (b = 0; b < (vector_size + condition_size); b++) {
                    unk_vec[b] += W[a * (vector_size + condition_size) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + condition_size) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            fprintf(fout_cxt, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout," %lf", unk_vec[b]);
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            if (model == 3) {
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout," %lf", unk_vec[b]);
                for (b = 0; b < (vector_size + condition_size); b++) fprintf(fout_cxt," %lf", unk_context[b]);
            }
            fprintf(fout,"\n");
            fprintf(fout_cxt,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fid2);
        fclose(fout);
        fclose(fout2);
        fclose(fout_cxt);
        fclose(fout2_cxt);
    }
    return 0;
}

/* Train model */
int train_ewe() {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin;
    real total_cost = 0;

    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if (fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    if (verbose > 0) fprintf(stderr, "condition size: %lld\n", condition_size);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    
    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    // Lock-free asynchronous SGD
    for (b = 0; b < num_iter; b++) {
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, ewe_thread, (void *)&thread_ids[a]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        free(thread_ids);

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, cost: %lf\n", time_buffer,  b+1, total_cost/num_lines);

        if (checkpoint_every > 0 && (b + 1) % checkpoint_every == 0) {
            fprintf(stderr,"    saving itermediate parameters for iter %03d...", b+1);
            save_params_return_code = save_params(b+1);
            if (save_params_return_code != 0)
                return save_params_return_code;
            fprintf(stderr,"done.\n");
        }

    }
    free(pt);
    free(lines_per_thread);
    return save_params(0);
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    vocab_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_T_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_cxt_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_T_cxt_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = malloc(sizeof(char) * MAX_STRING_LENGTH);
    int result = 0;
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
	printf("\t-write-header <int>\n");
	printf("\t\tIf 1, write vocab_size/vector_size as first line. Do nothing if 0 (default).\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\t-checkpoint-every <int>\n");
        printf("\t\tCheckpoint a  model every <int> iterations; default 0 (off)\n");
        result = 0;
    } else {
	if ((i = find_arg((char *)"-write-header", argc, argv)) > 0) write_header = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-cond", argc, argv)) > 0) condition_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        cost = malloc(sizeof(real) * num_threads);
        if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
        //if (model != 0 && model != 1) model = 2;
        if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
        else strcpy(vocab_file, (char *)"vocab.txt");
        if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
        else strcpy(save_W_file, (char *)"vectors");
        if ((i = find_arg((char *)"-save-word-cond-file", argc, argv)) > 0) strcpy(save_W_T_file, argv[i + 1]);
        else strcpy(save_W_T_file, (char *)"word_cond_vectors");
        // *** context vectors ***
        if ((i = find_arg((char *)"-save-context-file", argc, argv)) > 0) strcpy(save_W_cxt_file, argv[i + 1]);
        else strcpy(save_W_cxt_file, (char *)"vectors");
        if ((i = find_arg((char *)"-save-word-cond-context-file", argc, argv)) > 0) strcpy(save_W_T_cxt_file, argv[i + 1]);
        else strcpy(save_W_T_cxt_file, (char *)"word_cond_vectors");
        // *** context vectors ***

        if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
            strcpy(save_gradsq_file, argv[i + 1]);
            save_gradsq = 1;
        }
        else if (save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
        if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
        else strcpy(input_file, (char *)"cooccurrence.shuf.bin");
        if ((i = find_arg((char *)"-checkpoint-every", argc, argv)) > 0) checkpoint_every = atoi(argv[i + 1]);
        
        vocab_size = 0;
        fid = fopen(vocab_file, "r");
        if (fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
        fclose(fid);

        result = train_ewe();
        free(cost);
    }
    free(vocab_file);
    free(input_file);
    free(save_W_file);
    free(save_W_T_file);
    free(save_gradsq_file);
    return result;
}
