#!/bin/bash

##  Copyright (C) 2017 Huang Hengguan
##  hhuangaj [at] ust [dot] hk
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


set -e

nj=10
. cmd.sh
. path.sh


stage=0
## Configurable directories
train=data-fbank/train
dev=data-fbank/dev
test=data-fbank/test

#train=data/train_si84_noisy
#dev=data/dev_dt_05_noisy
#test=data/test_eval92_5k_noisy


lang=data/lang
gmm=exp/tri3
exp=exp/sru
lm=test_bg



export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=1
## tune learning rate
## Train





# Generate Alignments for Dev set
if [ $stage -le 0 ]; then

steps/align_fmllr.sh --nj "$nj" --cmd "$train_cmd" \
 data/dev data/lang exp/tri3 exp/tri3_ali_dev

fi



# Generate Fbank features
fbankdir=fbank

if [ $stage -le 1 ]; then
  mkdir -p data-fbank
  cd exp
  mkdir -p make_fbank
  cd ..
                                                                                                                                                                                               
  for x in train dev test; do         
    cp -r data/$x data-fbank/$x                                                                                                                                                           
    steps/make_fbank.sh --cmd "$train_cmd" --nj $nj data-fbank/$x exp/make_fbank/$x $fbankdir                                                                                                       
    steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir                                                                                                                                
  done     
fi



for lr in 0.25 ; do
    python3 steps_sru/train_sru.py $dev ${gmm}_ali_dev $train ${gmm}_ali $gmm ${exp}_$lr $lr


    ## Make graph
    [ -f $gmm/graph_${lm}/HCLG.fst ] || utils/mkgraph.sh ${lang}_${lm} $gmm $gmm/graph_${lm}

    ## Decode
    echo "tune acoustic scale"
    for ac in 0.1  ; do
    [ -f ${exp}_$lr/decode.done ] || bash steps_sru/decode_sru.sh --nj $nj --acwt $ac --scoring-opts "--min-lmwt 4 --max-lmwt 15"  \
        --add-deltas "true" --norm-vars "true" --splice-size "20" --splice-opts "--left-context=0 --right-context=4"  \
        $test $gmm/graph_${lm} ${exp}_$lr ${exp}_$lr/decode_$ac
    done
done

    
