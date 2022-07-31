## MedConQA: Medical Conversational Question Answering System based on Knowledge Graphs

###### 

### How to use

##### We offer three experience types:

*1. Triage*  You can take your symptoms as input, and we will triage it.

*2. Dagnosis* You can ask any medical questions

*3. Summary* Take the doctor-patient dialogue as input, and the medical record  will be output

##### We offer three input mode:

*1.interactive* Support human-computer interaction experience demo

*2.api* Provide API interface for direct call

*3.batch* Provide input files for batch processing



### Example

```
python main.py --mode api --type Dagnosis --message 我肚子好疼
```
<center><img src="image/Dagnosis.png" alt="img" style="zoom:50%;" /></center>
```
python main.py --mode interactiv --type Dagnosis
```
<center><img src="image/Triage.png" alt="img" style="zoom:50%;" /></center>
```
python main.py --mode batch --type Summary --file_name input.csv --result_file_name result.csv
```
<center><img src="image/Summary.png" alt="img" style="zoom:50%;" /></center>
