# SRU_Kaldi
## Installation

### Prerequisites

* Compiled Kaldi instance ([instructions](https://github.com/kaldi-asr/kaldi/blob/master/INSTALL))
* Install Anaconda and create the environment with python 3.7, pytorch 1.3.
* TIMIT data
## Major Scripts 

*`steps_sru/lib`: package for the implementations of the operations/NNs 
*`steps_sru/dataGenSequences*.py`:  data iteraters
*`steps_our/train*.py`: implementations of the SRU-HMM
*`steps_our/decode*.sh`:  decoder
*`steps_our/nnet-forward*.py`:  HMM state posterior probability estimator

## Usage

Quick intstructions for building SRU-HMM System:

(1) Move to Kaldi timit directory, e.g.,

`cd kaldi-trunk/egs/timit/s5`

(2) Copy all source code files (.sh, .py) and directories into current directory

(3) Specify TIMIT data directory in run_simple.sh e.g.,

 `timit=<your TIMIT directory>`

(4) Execute run_simple.sh

(5) Execute run_sru.sh

