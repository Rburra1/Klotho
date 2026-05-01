# Sample outputs

Trained on Tiny Shakespeare. 5,000 iterations, ~96 minutes on Apple M4 with MPS. Best val loss 1.49.

The model learned style and structure (speaker formatting, iambic cadence, archaic vocabulary, character names from across the corpus) without learning meaning. Expected ceiling for 10M params on 1MB of text.

## ROMEO seed

```
ROMEO:
I would thy love leave to come to see thee
And bring thee at thy lips and spirit
By this thing I will grand thy flower in what I
Shall be some office golden with thy worst.
```

## JULIET seed (temp 0.7)

```
JULIET:
So swear to him that a few best.

GREMIO:
What is that hast made, sir, and here, I think
He is not well-straight that Henry is the world
```

## To be seed (temp 0.6)

```
To be, and so straight the elders of the deed,
To make the foe, and at the coronation:
And, that it is not so, it is not a might on sweet
man, in the shipts of bound and whose thinking they be a content.

KING RICHARD II:
Sir Richard, are you france.
```
