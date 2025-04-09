# AmbedkarGPT (Fine-tuned GPT-2 / GPT-Neo)

## Setup (Windows Compatible)
```bash
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Steps
1. Add Ambedkar PDFs to `data/pdfs`
2. Extract text:
```bash
python extract_text.py
```
3. Preprocess:
```bash
python preprocess.py
```
4. Train:
```bash
python train.py
```
5. Test:
```bash
python test_model.py
```

## Notes
- Model: GPT-2 or GPT-Neo (125M)
- Runs on CPU (no_cuda=True)
- Best with small dataset first to test
