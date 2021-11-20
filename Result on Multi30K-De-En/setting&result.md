This model learn from scratch, and I didn't use Beam search for translation.

min_freq: 2

Epoch: 40

batch_size: 64

layer: 3 

emb_dim: 512

expan: 2 

POST_LN: True

DROPOUT = 0.1

label smoothing = 0.1

I saved not only the lowest val loss, after 50% of epoch, I saved 5 times when the epoch increased 10% (in this example, 24, 28, 32, 36, 40 had been saved)
and I calculated bleu score for all of those models.

bleu score:
'best val loss': 30.119620829294036

'epoch24': 30.710459472298986

'epoch28': 29.903834833565764

'epoch32': 30.761604382945123

'epoch36': 31.77815026971991 * best one

'epoch40': 31.02933102153035
