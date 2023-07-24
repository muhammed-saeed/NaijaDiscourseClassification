### Installation Guidance
Create conda environment


## installation of the our requirments


## installation of hte discopy

1- create two separate conda environments 
1.1 one with `python=3.7.15` and in this one clone both discopy 

```
conda create -n discopy python=3.7.15
git clone https://github.com/muhammed-saeed/discopy.git
git clone https://github.com/muhammed-saeed/discopy-data.git

```

I- install discopy-data repo first go to the discopy-data file and run

```
cd discopy-data
pip install -e .

```

II- install discopy repo second

```
cd discopy
pip install -e .

```

III- Download the specific release

`https://github.com/rknaebel/discopy/releases/download/1.1.0/bert-10.11.21-13.31.tar.gz`


IV- Test your model 

```
discopy-tokenize -i /home/CE/musaeed/output_dir/input_file.txt | discopy-add-parses -c | python3 /home/CE/musaeed/discopy/cli/bert/parse.py bert-base-uncased /home/CE/musaeed/bert_model/> /home/CE/musaeed/output_dir/output_file.txt
```
 IV- I have modified the following code to run  with our UI for the translation system it will run on port 8081 and this will use load_fast_text


```
https://github.com/muhammed-saeed/discopy/blob/master/app/run_bert_discourse_classifier_for_pcm.py
```
To Run the code do the following

```
python3 app/run_bert_discourse_classifier_for_pcm.py --model-path /home/CE/musaeed/discourse_cls/bert_model/ --port 8081 --reload

```
     

1.2. create a second conda environment and in this conda clone 

`conda create -n python=3.`

clone "discopy-vis" 
`https://github.com/muhammed-saeed/discopy-vis.git`

 
 1.2.1 install the requirements 
 `pip install -r requirements.txt `

1.2.2 uninstall werkzeug `pip uninstall werkzeug`
1.2.3 go to discopy-vis/app 
then run  python run.py --port 8000

```
discopy-tokenize -i /local/musaeed/discourse_classification/input_file.txt | discopy-add-parses -c | python3 /local/musaeed/discopy/cli/bert/parse.py bert-base-uncased /local/musaeed/discopy_models > /local/musaeed/discourse_classification/output_file.txt
```

## Treebank and annotation

1- cleaner the usc pcm train
this code modify the case where the naija train has fewer lines

```
python3 TreeBankAnnotated/scripts/cleaner.py

```
2- conllu dataset into csv and concatenate the trian/test/dev

```
python3 TreeBankAnnotated/scripts/conlluToCSV.py
python3 TreeBankAnnotated/scripts/concacteCSV.py
```

3- from the annotated data keep the explicit and implicit connectives


```
python3 TreeBankAnnotated/scripts/explicitImplicitOnly.py
```


4- then create the full text in english format

```
<!-- python3 TreeBankAnnotated/scripts/CreateFullTextEnglish.py -->
python3 TreeBankAnnotated/scripts/fullTextGeneratorAndMapping.py
```


5-create the corresponding NaijaText

```
python3 TreeBankAnnotated/scripts/fullTextMappingCodeToUse.py

```

5-1 we realized that the generated data looks the same a

```


```


