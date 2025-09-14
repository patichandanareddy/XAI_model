\# XAI\_model — Explainable Neural Networks for Constitutive Modeling



This repository contains experiments on using neural networks as surrogates for \*\*constitutive models of materials\*\* (elastoplasticity, hyperelasticity, and viscoelasticity), with a focus on \*\*explainability\*\* (permutation importance, saliency, integrated gradients).



---



\## 📂 Repository Structure



XAI\_model/

├── data/ # Datasets (Zenodo-provided)

│ ├── elastoplasticity/

│ ├── hyperelasticity/

│ └── viscoelasticity/

├── scripts/ # Training \& explanation scripts

│ ├── run\_elastoplastic.py

│ ├── run\_elastoplastic\_v2.py

│ ├── run\_hyperelastic.py

│ ├── run\_hyperelastic\_v2.py

│ ├── run\_viscoelastic\_v2.py

│ ├── convert\_hyperelastic\_tfrecords.py

│ ├── convert\_viscoelasticity\_tfrecords.py

│ ├── explain\_elasto.py

│ ├── explain\_hyper.py

│ └── explain\_visco.py

├── outputs/ # Model checkpoints, plots, explanations

│ ├── elastoplastic\_v2/

│ ├── hyperelastic\_v2/

│ └── viscoelastic\_v2/

└── requirements.txt



---



\## ⚙️ Environment Setup



```bat

conda create -n xai-torch python=3.10

conda activate xai-torch



pip install torch torchvision torchaudio

pip install tensorflow keras scikit-learn matplotlib tqdm



Workflows

1\. Elastoplasticity

Training

python scripts\\run\_elastoplastic\_v2.py ^

&nbsp; --data\_dir "data\\elastoplasticity" ^

&nbsp; --epochs 50 ^

&nbsp; --batch 4096 ^

&nbsp; --threads 8 ^

&nbsp; --standardize ^

&nbsp; --max\_files 500 ^

&nbsp; --limit 400000 ^

&nbsp; --sched\_factor 0.5 ^

&nbsp; --sched\_patience 3 ^

&nbsp; --sched\_min\_lr 1e-6 ^

&nbsp; --early\_stop ^

&nbsp; --early\_stop\_patience 10





Outputs:



outputs/elastoplastic\_v2/model.pt



outputs/elastoplastic\_v2/loss\_curve.png



outputs/elastoplastic\_v2/predictions\_onefile.png



outputs/elastoplastic\_v2/residuals.png



Explainability

python scripts\\explain\_elasto.py ^

&nbsp; --data\_dir "data\\elastoplasticity" ^

&nbsp; --model\_path "outputs\\elastoplastic\_v2\\model.pt" ^

&nbsp; --out\_dir "outputs\\elastoplastic\_v2" ^

&nbsp; --standardize ^

&nbsp; --use\_params ^

&nbsp; --target\_delta ^

&nbsp; --max\_files 300 ^

&nbsp; --limit 300000





Artifacts:



explain\_permutation.png



explain\_saliency.png



explain\_integrated\_gradients.png



explain\_summary.json



2\. Hyperelasticity

Convert TFRecords → NPY

python scripts\\convert\_hyperelastic\_tfrecords.py --root "data\\hyperelasticity"



Training

python scripts\\run\_hyperelastic\_v2.py ^

&nbsp; --data\_dir "data\\hyperelasticity" ^

&nbsp; --epochs 30 ^

&nbsp; --batch 2048 ^

&nbsp; --threads 8 ^

&nbsp; --standardize ^

&nbsp; --limit 500000



Explainability

python scripts\\explain\_hyper.py ^

&nbsp; --data\_dir "data\\hyperelasticity" ^

&nbsp; --model\_path "outputs\\hyperelastic\_v2\\model.pt" ^

&nbsp; --out\_dir "outputs\\hyperelastic\_v2" ^

&nbsp; --standardize ^

&nbsp; --max\_files 300 ^

&nbsp; --limit 300000



3\. Viscoelasticity

Convert TFRecords → NPY

python scripts\\convert\_viscoelasticity\_tfrecords.py --root "data\\viscoelasticity"



Training

python scripts\\run\_viscoelastic\_v2.py ^

&nbsp; --data\_dir "data\\viscoelasticity" ^

&nbsp; --epochs 20 ^

&nbsp; --batch 4096 ^

&nbsp; --threads 8 ^

&nbsp; --standardize ^

&nbsp; --max\_files 500 ^

&nbsp; --limit 400000



Explainability

python scripts\\explain\_visco.py ^

&nbsp; --data\_dir "data\\viscoelasticity" ^

&nbsp; --model\_path "outputs\\viscoelastic\_v2\\model.pt" ^

&nbsp; --out\_dir "outputs\\viscoelastic\_v2" ^

&nbsp; --standardize ^

&nbsp; --target\_delta ^

&nbsp; --batch 4096 ^

&nbsp; --permutation\_repeats 3 ^

&nbsp; --max\_files 300 ^

&nbsp; --limit 300000



✅ Results (Summary)



Elastoplasticity v2



R² (all): ~0.999



Separate R²: Elastic ~0.9994, Plastic ~0.9988



Hyperelasticity v2



Loss ~1e-4 after 30 epochs



Predictions closely match stress–strain curves



Viscoelasticity v2



Final train loss: ~5.6e-6



Explanations highlight time-dependence (viscous effects)

