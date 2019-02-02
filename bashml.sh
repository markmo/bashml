#!/usr/bin/env bash

# Implements Logistic Regression for both numerical and categorical features. Applies feature scaling to
# numerical features.
# 
# Where delimiter characters can be embedded within quoted fields, preprocess the file first, replacing
# the delimiter character with an alternative delimiter that doesn't occur within fields; e.g.
#
#     ./bashml.sh -p -s ; -f $1
#
# Preprocessing produces a file named "raw.csv".
#
# Training splits the input file into "train.csv" and "test.csv" according to the `test_ratio` argument; e.g.
#
#     ./bashml.sh -t -n 6,7,9 -c 2,4,11 -y 1 -r .2 --header -d ';' -f raw.csv -v 1 -e $1
#
# Make sure you specify the replacement delimiter as the new delimiter character. See usage for more
# details:
#
#     ./bashml.sh -h
#
# In the above example, the script is in training mode `-t`, numerical features from columns 6, 7 and 9
# (zero-based indexing) are specified, categorical features from columns 2, 4 and 11, the y label is
# taken from column 1, a test split of 0.2 is set, the input file includes a header line, the column
# delimiter is the character ";", the input filename is "raw.csv", verbosity of reporting is set to level 1,
# and the number of epochs to train for is specified using `-e`.

UNK='<UNK>'  # unknown token (categorical feature value)


# Initialize a matrix with zero values
zeros() {
    local -n A=$1    # matrix to initialize, shape [p,q]
    local p=$2       # number rows
    local q=$3       # number cols
    local i
    local j
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            A[$i,$j]=0
        done
    done
}


# Calculate the dot product of two matrices
dot() {
    local -n A=$1    # first matrix, shape [p,q]
    local -n B=$2    # second matrix, shape [q,r]
    local -n C=$3    # result matrix, shape [p,r]
    local p=$4       # number rows in A
    local q=$5       # number cols in A, rows in B
    local r=$6       # number cols in B
    local i
    local j
    local k
    local sum
    for ((i=0; i<p; i++)) do
        for ((j=0; j<r; j++)) do
            sum=0
            for ((k=0; k<q; k++)) do
                sum=$(bc -l <<< "$sum + ${A[$i,$k]} * ${B[$k,$j]}")
            done
            C[$i,$j]=$sum
        done
    done
}


# Calculate the logistic function
sigmoid() {
    local -n A=$1    # input, shape [p,q]
    local -n B=$2    # output, shape [p,q]
    local p=$3       # number rows
    local q=$4       # number cols
    local i
    local j
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            B[$i,$j]=$(bc -l <<< "1 / (1 + e(-1 * ${A[$i,$j]}))")
        done
    done
}


# Subtract a vector from a matrix
subtract_vec_from_mat() {
    local -n A=$1    # matrix, shape [p,q]
    local -n b=$2    # vector, shape [p]
    local -n C=$3    # matrix, shape [p,q]
    local p=$4       # number rows
    local q=$5       # number cols
    local i
    local j
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            C[$i,$j]=$(bc -l <<< "${A[$i,$j]} - ${b[$i]}")
        done
    done
}


# Calculate loss
loss() {
    local -n A=$1    # h, shape [p,q]
    local -n b=$2    # y, shape [p]
    local p=$3       # number rows (examples)
    local q=$4       # number cols (classes)
    local i
    local j
    local sum=0
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            sum=$(bc -l <<< "$sum + (-${b[$i]} * l(${A[$i,$j]}) - (1 - ${b[$i]}) * l(1 - ${A[$i,$j]}))")
        done
    done
    printf "Loss: %.5f\n" $(bc -l <<< "$sum / $p")
}


# Calculate gradients
gradient() {
    local -n A=$1    # X, shape [p,q]
    local -n B=$2    # h_minus_y, shape [p,r]
    local -n C=$3    # grads, shape [q,r]
    local p=$4       # number rows in A,B
    local q=$5       # number cols in A
    local r=$6       # number cols in B
    local i
    local j
    local k
    local sum
    for ((i=0; i<q; i++)) do
        for ((j=0; j<r; j++)) do
            sum=0
            for ((k=0; k<p; k++)) do
                sum=$(bc -l <<< "$sum + ${A[$k,$i]} * ${B[$k,$j]}")
            done
            C[$i,$j]=$(bc -l <<< "$sum / $p")
        done
    done
}


# Update weights
update() {
    local -n A=$1    # grads, shape [q,p]
    local -n B=$2    # theta, shape [p,q]
    local p=$3       # number rows (features)
    local q=$4       # number cols (classes)
    local lr=$5      # learning rate
    local i
    local j
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            B[$i,$j]=$(bc -l <<< "${B[$i,$j]} - $lr * ${A[$i,$j]}")
        done
    done
}


# fit model to training data
fit() {
    local -n X=$1    # input data X, shape [m,n]
    local -n y=$2    # labels, shape [m]
    local m=$3       # number rows in X (examples)
    local n=$4       # number cols in X (features)
    local c=$5       # number classes
    local e=$6       # number epochs
    local lr=$7      # learning rate
    local v=$8       # verbose mode: 0 - no reporting, 1 - loss only, 2 - debug info
    local -A W       # weights (theta), shape [n,c]
    local -A Z       # dot product, shape [m,c]
    local -A H       # shape [m,c]
    local -A D       # (H - y) shape [m,c]
    local -A G       # gradients, shape [n,c]
    local i
    local j

    if [[ $v -gt 1 ]]; then
        echo ''
        echo 'y:'
        print_vec y
    fi

    # initialize weights matrix with zeros
    zeros W $n $c
    print_mat W $n $c 'W:'

    for ((i=1; i<=e; i++)) do
        [[ $v -gt 0 ]] && echo "Epoch $i"
        dot X W Z $m $n $c
        print_mat Z $m $c 'Z:'
        
        sigmoid Z H $m $c
        print_mat H $m $c 'H:'
        
        subtract_vec_from_mat H y D $m $c
        print_mat D $m $c 'H - y:'
        
        gradient X D G $m $n $c
        print_mat G $n $c 'G:'
        
        update G W $n $c $lr
        print_mat W $n $c 'W:'
        
        [[ $v -gt 0 ]] && loss H y $m $c
    done

    loss H y $m $c

    save_model W $n $c 'weights.txt'
}

predict() {
    local -n X=$1    # input data X, shape [m,n]
    local -n y=$2    # labels, shape [m]
    local -n W=$3    # weights (theta)
    local m=$4       # number rows in X (examples)
    local n=$5       # number cols in X (features)
    local c=$6       # number classes
    local v=$7       # verbose mode
    local -A Z       # dot product, shape [m,c]
    local -A H       # shape [m,c]
    local i
    local p          # prediction
    local k=0        # number correct
    dot X W Z $m $n $c
    sigmoid Z H $m $c
    for ((i=0; i<m; i++)) do
        for ((j=0; j<c; j++)) do
            p=$(bc -l <<< "${H[$i,$j]} > 0.5")
            if [[ $v -gt 1 ]]; then
                printf "y_true:${y[$i]}, y_pred:$p (%.4f)\n" "${H[$i,$j]}"
            fi
            if [[ ${y[$i]} -eq p ]]; then
                k=$((k + 1))
            fi
        done
    done
    printf "Accuracy: %.2f\n" $(bc -l <<< "$k / ($m * $c)")
}

save_model() {
    local -n A=$1    # weights (theta)
    local n=$2       # number features
    local c=$3       # number classes
    local o="$4"     # model filename
    local i
    local j
    local line
    if [[ -f "$o" ]]; then
        rm $o
    fi
    for ((i=0; i<n; i++)) do
        line=''
        for ((j=0; j<c; j++)) do
            [[ $j -gt 0 ]] && line+=$'\t'
            line+="${A[$i,$j]}"
        done
        echo "$line" >> "$o"
    done
}

load_model() {
    local -n A=$1    # weights (theta)
    local -n p=$2    # number features
    local -n q=$3    # number classes
    local f="$4"     # model filename
    local -a r       # row of values
    local i=0
    local j
    local k
    local line
    k=0
    while read line; do
        IFS=$'\t' read -d '' -r -a r <<< "$line"
        if [[ $k -eq 0 ]]; then
            k=${#r[@]}
        fi
        for ((j=0; j<k; j++)) do
            A[$i,$j]="${r[$j]}"
        done
        i=$((i + 1))
    done < "$f"
    p=$i
    q=$k
}


# convert categorical features into numeric features
onehot_encode() {
    local -n a=$1       # array of categorical feature values, shape [p]
    local -n E=$2       # onehot encoded matrix, shape [p,q]
    local -n v=$3       # dictionary of index by category
    local n=${#v[@]}    # vocab size
    local p=${#a[@]}    # number rows (examples)
    local q             # number cols (categories)
    local l             # label
    local i
    local j
    local val
    local idx
    local k=1
    if [[ $n -eq 0 ]]; then
        echo 'Creating vocab'
        v=()
        v[$UNK]=0
        for l in "${a[@]}"; do
            if [[ -n "$l" && -z "${v[$l]}" ]]; then
                v[$l]=$k
                k=$((k + 1))
            fi
        done
    fi
    # for i in "${!v[@]}"; do
    #     echo "${i}=${v[$i]}"
    # done
    q=${#v[@]}
    for ((i=0; i<p; i++)) do
        for ((j=0; j<q; j++)) do
            val=$([[ -z "${a[$i]}" ]] && echo "$UNK" || echo "${a[$i]}")
            idx=${v[$val]}
            if [[ $j -eq idx ]]; then
                E[$i,$j]=1
            else
                E[$i,$j]=0
            fi
        done
    done
}


# calculate mean of set of values
calc_mean() {
    local -n a=$1       # array of numeric values
    local n=${#a[@]}    # number values
    local i
    local val
    local sum=0
    for ((i=0; i<n; i++)) do
        val=$([[ -z "${a[$i]}" ]] && echo '0' || echo "${a[$i]}")
        sum=$(bc -l <<< "$sum + $val")
    done
    echo $(bc -l <<< "$sum / $n")
}


# calculate standard deviation of set of values
calc_std() {
    local -n a=$1       # array of numeric values
    local mu=$2         # mean of array values
    local n=${#a[@]}    # number values
    local i
    local val
    local sum=0
    for ((i=0; i<n; i++)) do
        val=$([[ -z "${a[$i]}" ]] && echo "$mu" || echo "${a[$i]}")
        sum=$(bc -l <<< "$sum + ($val - $mu)^2")
    done
    echo $(bc -l <<< "sqrt($sum / $n)")
}


load_numeric_features() {
    local -n A=$1       # input data X, shape [p,q]
    local -n b=$2       # indices of numeric feature columns
    local s=$3          # start index for feature cols
    local f="$4"        # filename
    local d="$5"        # delimiter of feature cols
    local h=$6          # header indicator
    local -a c          # temp col array
    local p             # number rows
    local q=${#b[@]}    # number cols
    local i
    local j
    local k
    local mu
    local sigma
    local val
    echo "Loading numerical features [${b[@]}]"
    for ((j=0; j<q; j++)) do
        k=$((b[$j] + 1))
        # awk does not support look-ahead or look-behind
        # IFS=$'\n' read -d '' -r -a c <<< $(awk -F "\\\"*$d(?![^\\\"]*?\\\"$d)\\\"*" "{print (\$${k} == \"\" ? 0 : \$${k})}" "$f")
        IFS=$'\n' read -d '' -r -a c <<< $(awk -F "\\\"*$d\\\"*" "{print (\$${k} == \"\" ? 0 : \$${k})}" "$f")
        if [[ $h = true ]]; then
            c=("${c[@]:1}")
        fi
        p=${#c[@]}
        mu=$(calc_mean c)
        sigma=$(calc_std c $mu)
        for ((i=0; i<p; i++)) do
            val=$([[ -z "${c[$i]}" ]] && echo "$mu" || echo "${c[$i]}")
            # scale the feature value: between -1 and 1, centred on 0
            A[$i,$((s + j))]=$(bc -l <<< "($val - $mu) / $sigma")
        done
    done
}


load_categorical_features() {
    local -n A=$1       # input data X, shape [p,..]
    local -n b=$2       # indices of categorical feature columns
    local s=$3          # start index for feature cols
    local f="$4"        # filename
    local d=$5          # delimiter of feature cols
    local h=$6          # header indicator
    local p=$7          # number rows
    local -n q=$8       # number cols
    local t=$9          # test mode: load vocabs from file
    local -a c          # temp col array, shape [p]
    local -A enc        # temp onehot encoded matrix, shape [p,..]
    local -A voc        # vocabulary of unique categories
    local r             # number categories
    local i
    local j
    echo "Loading categorical features [${b[@]}] ($([[ $t = true ]] && echo 'prediction' || echo 'training'))"
    q=0
    for k in "${b[@]}"; do
        k=$((k + 1))
        IFS=$'\n' read -d '' -r -a c <<< $(awk -F "\\\"*$d\\\"*" "{print (\$$k == \"\" ? \"$UNK\" : \$$k)}" "$f")
        if [[ $h = true ]]; then
            c=("${c[@]:1}")
        fi
        # trim newlines from column values (that exist on last column)
        for ((i=0; i<p; i++)) do
            c[$i]=$(echo "${c[$i]}" | tr -d '\r')
        done
        voc=()
        if [[ $t = true ]]; then
            echo "Loading vocab file: vocab.$((k - 1)).txt"
            load_vocab voc "vocab.$((k - 1)).txt"
        fi
        onehot_encode c enc voc
        # echo ''
        # echo "Vocab $((k - 1)):"
        # for i in ${!voc[@]}; do
        #     echo "${i}=${voc[$i]}"
        # done
        if [[ $t = false ]]; then
            save_vocab voc "vocab.$((k - 1)).txt"
        fi
        r=${#voc[@]}
        q=$((q + r))
        # print_mat enc $p $r 'Encoded:'
        for ((i=0; i<p; i++)) do
            for ((j=0; j<r; j++)) do
                A[$i,$((s + j))]=${enc[$i,$j]}
            done
        done
        s=$((s + r))
    done
}


load_vocab() {
    local -n V=$1       # vocab dict, token=idx
    local f=$2          # vocab filename
    local line
    local i=0
    while read line; do
        V[$line]=$i
        i=$((i + 1))
    done < $f
}

save_vocab() {
    local -n V=$1       # vocab dict, token=idx
    local o=$2          # output filename
    local -A R          # reversed dict, idx=token
    local n=${#V[@]}    # vocab size
    local t             # each token
    local i             # each index
    # flip dict
    for t in ${!V[@]}; do
        R[${V[$t]}]=$t
    done
    if [[ -f $o ]]; then
        rm $o
    fi
    for ((i=0; i<n; i++)) do
        echo ${R[$i]} >> $o
    done
}

load_labels() {
    local -n a=$1     # labels array
    local f=$2        # filename
    local d=$3        # delimiter of feature cols
    local h=$4        # header indicator
    local i=$5        # label col index
    IFS=$'\n' read -d '' -r -a a <<< $(awk -F "\\\"*$d\\\"*" "{print \$$((i + 1))}" "$f")
    if [[ $h = true ]]; then
        a=("${a[@]:1}")
    fi
}


csv_to_arr() {
    local -n a=$1     # indices as array
    local b=$2        # indices as csv
    IFS=',' read -r -a a <<< "$b"
}

preprocess() {
    local f=$1        # filename
    local d=$2        # delimiter
    local r=$3        # replacement delimiter
    local o=$4        # output filename
    sed "s/$d/$r/g" $f | sed "s/\(\".*\)$r\(.*\"\)/\1$d\2/g" > $o
}

# shuffle rows for training
shuf() {
    local h=$1    # header indicator
    shift
    local i
    [[ $h = true ]] && i=1 || i=0
    awk -v i="$i" 'BEGIN {srand(); OFMT="%.17f"} {if (NR!=i) {print rand(), $0}}' "$@" |
        sort -k1,1n | cut -d ' ' -f2-;
}

# determine split point to split training and test sets
split() {
    local f=$1    # filename
    local r=$2    # test ratio
    local h=$3    # header indicator
    local k       # line count of file
    local s       # number lines of test set
    k=$(wc -l < $f | tr -d ' ')  # delete whitespace
    [[ $h = true ]] && k=$((k - 1))
    s=$(printf %.0f $(bc <<< "$k * $r"))  # round to int
    echo $s
}

save_test_set() {
    local s=$1    # split point
    shift
    if [[ $s -gt 0 ]]; then
        head -n $s "$1" > test.csv
    fi
}

save_training_set() {
    local s=$1    # split point
    shift
    tail -n "+$((s + 1))" "$1" > train.csv
}

usage() {
    echo ''
    echo 'Usage:'
    echo './bashml.sh ...args'
    echo ''
    echo 'Arguments:'
    echo '-f | --file            : data file'
    echo '-o | --pp_file         : preprocessed file'
    echo '-d | --delimiter       : column delimiter'
    echo '-s | --repl_delim      : replacement column delimiter'
    echo '-a | --header          : file has header'
    echo '-n | --numer_feat_idxs : numerical feature indices'
    echo '-c | --categ_feat_idxs : categorical feature indices'
    echo '-y | --y_i             : column index of output variable'
    echo '-l | --learning_rate   : learning rate'
    echo '-e | --n_epochs        : number epochs'
    echo '-r | --test_ratio      : test ratio'
    echo '-t | --train           : training mode'
    echo '-p | --preprocess      : preprocess data file'
    echo '-v | --verbose         : print debug info'
    echo '-h | --help            : show this usage information'
    echo ''
}

# Hyperparameters
filename=
pp_filename='raw.csv'
delimiter=','
repl_delim=';'
header=false
numer_feat_ixs_csv=
categ_feat_ixs_csv=
y_i=
learning_rate=0.01
n_epochs=2
do_train=false
do_preprocess=false
test_ratio=0.1
split_count=0
n_classes=1
verbose=1

while [[ "$1" != '' ]]; do
    case $1 in
        -f | --file )               shift
                                    filename=$1
                                    ;;
        -o | --pp_file )            shift
                                    pp_filename=$1
                                    ;;
        -d | --delimiter )          shift
                                    delimiter=$1
                                    ;;
        -s | --repl_delim )         shift
                                    repl_delim=$1
                                    ;;
        -a | --header )             header=true
                                    ;;
        -n | --numer_feat_ixs )     shift
                                    numer_feat_ixs_csv=$1
                                    ;;
        -c | --categ_feat_ixs )     shift
                                    categ_feat_ixs_csv=$1
                                    ;;
        -y | --y_i )                shift
                                    y_i=$1
                                    ;;
        -l | --learning_rate )      shift
                                    learning_rate=$1
                                    ;;
        -e | --n_epochs )           shift
                                    n_epochs=$1
                                    ;;
        -r | --test_ratio )         shift
                                    test_ratio=$1
                                    ;;
        -t | --train )              do_train=true
                                    ;;
        -p | --preprocess )         do_preprocess=true
                                    ;;
        -v | --verbose )            shift
                                    verbose=$1
                                    ;;
        -h | --help )               usage
                                    exit
                                    ;;
        * )                         usage
                                    exit 1
    esac
    shift
done


# Print vector
print_vec() {
    if [[ $verbose -gt 1 ]]; then
        local -n a=$1
        echo ${a[@]}
    fi
}

# Print matrix
print_mat() {
    if [[ $verbose -gt 1 ]]; then
        local -n M=$1     # matrix, shape [p,q]
        local p=$2        # number rows
        local q=$3        # number cols
        local title=$4
        local i
        local j
        local v
        # for i in ${!M[@]}; do
        #     echo "${i}=${M[$i]}"
        # done
        printf '%0.s-' {1..55}
        printf '\n'
        if [[ -z "$title" ]]; then
            echo 'Matrix:'
        else
            echo $title
        fi
        for ((i=0; i<p; i++)) do
            for ((j=0; j<q; j++)) do
                [[ $j > 0 ]] && printf '\t'
                v=${M[$i,$j]}
                if [[ -z "$v" ]]; then
                    break
                fi
                printf "%f" $v
            done
            printf '\n'
        done
        printf '%0.s-' {1..55}
        printf '\n'
    fi
}

prepare_data() {
    local -n X=$1              # input data, shape [m,n]
    local -n y=$2              # labels, shape [m]
    local -n p=$3              # number rows (examples)
    local -n q=$4              # number cols (features)
    local f=$5                 # filename
    local h=$6                 # header indicator
    local t=$7                 # test mode: load vocabs from file
    
    # Globals
    local -a numer_feat_ixs
    local -a categ_feat_ixs
    local n_numer_feats
    local n_categ_feats
    local n_categ_cols

    load_labels y $f $delimiter $h $y_i
    m=${#y[@]}
    echo 'm:' $m
    # echo ${y[@]}
    
    csv_to_arr numer_feat_ixs $numer_feat_ixs_csv
    n_numer_feats=${#numer_feat_ixs[@]}
    # echo 'n_numer_feats:' $n_numer_feats
    # echo 'numer_feat_ixs:' "(${numer_feat_ixs[@]})"
    
    csv_to_arr categ_feat_ixs $categ_feat_ixs_csv
    n_categ_feats=${#categ_feat_ixs[@]}
    # echo 'n_categ_feats:' $n_categ_feats
    # echo 'categ_feat_ixs:' "(${categ_feat_ixs[@]})"
    
    load_numeric_features X numer_feat_ixs 0 $f $delimiter $h
    # echo 'm:' ${#X[@]}
    q=${#numer_feat_ixs[@]}
    # echo 'n:' $q
    
    load_categorical_features X categ_feat_ixs $n $f $delimiter $h $p n_categ_cols $t
    q=$((q + n_categ_cols))
    echo 'n:' $q

    print_mat X $(($p<10?$p:10)) $q 'X:'
}

train() {
    local f=$1                 # filename
    local h=$2                 # header indicator
    local -A X_train           # input data, shape [m,n]
    local -a y_train           # labels, shape [m]
    local m                    # number examples
    local n                    # number features
    prepare_data X_train y_train m n $f $h false
    time fit X_train y_train $m $n $n_classes $n_epochs $learning_rate $verbose
}

test() {
    local f=$1                 # filename
    local h=$2                 # header indicator
    local -A X_test            # input data, shape [m,n]
    local -a y_test            # labels, shape [m]
    local -A weights           # model weights
    local m                    # number rows (examples)
    local n                    # number cols (features)
    local c                    # number classes
    prepare_data X_test y_test m n $f $h true
    load_model weights n c 'weights.txt'
    print_mat weights $n $c 'W:'
    predict X_test y_test weights $m $n $n_classes $verbose
}

if [[ $do_preprocess = true ]]; then
    preprocess $filename $delimiter $repl_delim $pp_filename
elif [[ $do_train = true ]]; then
    echo 'Prepare data...'
    s=$(split $filename $test_ratio $header)
    echo "split point: $s"

    # TODO - training set output getting truncated
    # shuf $header $filename | tee >(save_test_set $s) >(save_training_set $s)
    # Workaround:
    shuf $header $filename > 'shuffled.txt'
    save_test_set $s 'shuffled.txt'
    save_training_set $s 'shuffled.txt'

    echo ''
    echo 'Training...'
    train 'train.csv' false
else
    echo 'Predicting...'
    test 'test.csv' false
fi
