A pytorch version of Transformer, based on Attention is all you need.

In this model, Batch_First = True!!!

If use this model please fix the sequence length in the Field (e.g., src: fix_length = fix_len, tgt: fix_length = fix_len + 1)

Because the teach forcing learning pattern, we use loss_criterion(mod(src, tgt[:-1]), tgt[1:])

the test dataset for this model is Multi30k, and this model achieved bleu score: 32+ (I didn't do too many hyper parameters-tuning)
the result will be added soon
