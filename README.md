# granularitySearch<br>
keywords: feature selection, granularity, attribute reduction<br>
## About theme of this Repo
  The theme this repo focused on is the filter feature selection method for discrete data. Rough sets theory is one of effecitive tool to imprecise, incomplete, uncertainty information processing. We have developed two efficient reduction algorithms for inconsistent decision tables.  The proposed algorithms, i.e., GS and GSV, are extensions of QGARA-FS and QGARA-BS proposed in [3], and heuristic function or acceleration strategy used in GS and GSV, i.e., granualarity approximation, is  the extension of positive approximation proposed in [1]. 
## How to use
  Make sure that your data are stored in 'file.txt' and its content are consistent with "the vlaue of object 1 for feature 1, the vlaue of object 1 for feature 2,...., the value of object 1 for decision feature". Then run codes as follows:
  ```python
  reducer1=Reducer('file.txt')
  red_pos=reducer1.GS('POS')# positive region preservation reduction
  red_GED=reducer1.GS('GED')# general decision preservation reduction
  red_DIS=reducer1.GS('DIS')# distribution preservation reduction 
  red_MDS=reducer1.GS('MDS')# maximum distribution preservation reduction 
  red_IND=reducer1.GS('IND')# relative indiscernibility preservation reduction
  ```
  The output of GS is the features selected by evaluation functions, and roughly speaking, it preserves the same classification ability as entire attributes set of original data.

bibiliography:<br>
  [1] Qian Y, Liang J, Pedrycz W, et al. Positive approximation: An accelerator for attribute reduction in rough set theory[J]. Artificial Intelligence, 2010, 174(9):597-618.<br>
  [2] Li M, Shang C, Feng S, et al. Quick attribute reduction in inconsistent decision tables[J]. Information Sciences An International Journal, 2014, 254:155-180.<br>
  [3] Ge Hao , et al. Quick general reduction algorithms for inconsistent decision tables[J]. International Journal of Approximate Reasoning, 2017, 82:56-80.<br>
