# `trash` training images

Put images of **trash** items in this folder (no sub-folders).

**What belongs here:** Anything genuinely non-recyclable: crisp packets, dirty nappies, cigarette butts, polystyrene, mixed-material packaging, sweet wrappers.

**Where to find images:** TrashNet `trash` (~130 images - the smallest class, top it up yourself).

## Rules of thumb

* Accepted file types: .bmp, .jpeg, .jpg, .png, .webp.
* Aim for **at least 100 images per class**; 20 is the absolute floor and will overfit.
* Vary background, lighting, angle and distance. A hundred photos of the same bottle on
  the same table teaches the model about your table, not about plastic.
* One dominant object per photo. If two categories are visible, pick the one that fills
  the frame or drop the image.
* Roughly balance the classes. `ml/prepare_dataset.py` prints the imbalance ratio and
  `ml/train.py` applies class weights, but neither can invent data.
* Keep the split honest: do not put near-duplicate shots of the same object in here and
  expect the val score to mean anything - the splitter is random, so duplicates leak.

Then run, from the repository root:

    python ml/prepare_dataset.py --dry-run
    python ml/prepare_dataset.py
    python ml/train.py
