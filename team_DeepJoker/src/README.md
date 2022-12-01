## Training

Before training, you need to create the config folder. After training, the config folder will be saved.

Train a model for task 1

```python
mkdir config
python train.py --task 1
```

Train a model for task 2

```python
python train.py --task 2
```

## Evaluation

```python
mkdir submission
python main.py --data-type validation --submission-path ./submission --task 1
python main.py --data-type validation --submission-path ./submission --task 2
```

The commands above will output their predictions to a submission folder

## Team info :

Team name: DeepJoker

Team participants: Sijiang Li, Jian Kang

Emails: 20215228067@stu.suda.edu.cn, jiankang@suda.edu.cn

Countr(y/ies): 1
