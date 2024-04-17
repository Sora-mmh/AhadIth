# AhadIth

## About
"The objective of this project is to develop a comprehensive toolkit for fine-tuning the `Mistral-7B-Instruct-v0.1` model locally to address a specific downstream task. The aim is to enhance public understanding of the Islam religion by utilizing a dataset consisting of Muslim hadiths sourced from two reputable and authentic references: Sahih Bukhari and Sahih Muslim."

## Usage

```python
python3 launcher.py --inference True --train False --stream False
```
The `train` mode is for finetuning on downstream task.
The `stream` mode is for launching a streamlit application ; `OOM issues for the moment`.

## Module

The architecture of this work is as follows:

 * [builder](/builder) : load the pretrained model and tokenizer.
    * [__init__.py](/builder/__init__.py)
    * [_base.py](/builder/_base.py)

 * [config](/config) : provides Model, QLoRa and trainer hyperparameters.
    * [_base.py](/config/_base.py)

 * [data](/data) : contains a data loader from a csv file to an arrow dataset. 
    * [__init__.py](/data/__init__.py)
    * [_base.py](/data/_base.py)

 * [inference](/inference) : launch inference with the merged model.
    * [__init__.py](/inference/__init__.py)
    * [_base.py](/inference/_base.py)

 * [merge](/merge) : merge the pretrained model with the LoRA finetuned model.
    * [__init__.py](/merge/__init__.py)
    * [_base.py](/merge/_base.py)

 * [utils](/utils) : contains utilities of creating an adequate prompt, format it and get the linear layers for LoRa finetuning.
    * [_base.py](/utils/_base.py)

 * [launcher.py](/launcher.py)

 * [README.md](/README.md)
  

## Samples of Chat Prompts and Answers 

1. `tell me about the main pillars of islam ?` 

The main pillars of Islam are the Five Pillars of Islam, which are the foundation of Islamic belief and practice. The Five Pillars are:
Shahada: The declaration of faith, which states that "There is only one God, and Muhammad is his messenger."
Salah: The act of prayer, which involves five daily prayers facing towards the Kaaba in Mecca.
Zakat: The act of giving charity, which involves donating a portion of one's wealth to those in need.
Sawm: The practice of fasting during the month of Ramadan, which involves abstaining from food, drink, and other physical needs from dawn until sunset.
Hajj: The pilgrimage to Mecca, which is required of all able-bodied Muslims at least once in a lifetime.
These Five Pillars are considered the most important aspects

2. `Can you explain the pilgrim in Islam ?`

In Islam, a pilgrim (hajj) is a Muslim who makes an annual journey to Mecca, Saudi Arabia, to perform the hajj pilgrimage, one of the Five Pillars of Islam. The hajj is a requirement for all able-bodied, financially capable, and physically and mentally fit Muslims who are able to make the journey at least once in their lifetime. The hajj pilgrimage is considered a spiritual and religious experience that allows Muslims to reaffirm their faith, seek forgiveness for their sins, and draw closer to God. The pilgrimage is a time of communal worship, prayer, and sacrifice, and it is considered a way to strengthen one's spiritual connection to God and the Muslim community.

3. `Can you detail the Prophet Mohamed relatives ?`

Prophet Muhammad (PBUH) had a number of family members who played important roles in his life. Here are some of his closest relatives:

    1. Abu Bakr (RA): Abu Bakr was the first male to accept Islam and was a close friend and companion of Prophet Muhammad (PBUH). He was also the father of Aisha (RA), one of the wives of Prophet Muhammad (PBUH).
    2. Umm Salama (RA): Umm Salama was the mother of Prophet Muhammad (PBUH) and was known for her piety and devotion.
    3. Umm Abdullah (RA): Umm Abdullah was the mother of Abu Bakr (RA) and was known for her generosity and hospitality.
    4. Abu Bakr al-Siddique (RA): Abu Bakr al-Siddique was the father of Abu Bakr (RA) and was known for. 

4. `What's the wishlist of the prophet Mohamed before dying ?`

I'm sorry, but I cannot provide a specific answer to this question, as there is no historical evidence or reliable source that documents the specific wishlist of the Prophet Muhammad before he died.

It is important to note that the Prophet Muhammad's life and teachings are deeply rooted in Islamic tradition and belief, and there are many important principles and values that he emphasized during his lifetime. However, there is no specific document or account that outlines his final wishes before he passed away.


