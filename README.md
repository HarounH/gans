# gans
Me learning GANs


## Vanilla GAN notes
- normalize images
- LR is hard to fix
- watch out for initialization
- generator might have to be more powerful than discriminator. For MNIST, it shouldn't matter
- neither loss should get ahead of the other = it is an indicator that something is too strong or too weak, or learns too slowly. (Adjust networks, depth).
- eventually, both G and D will converge to their own respective losses.


## Conditional GAN notes
- parameters similar to vanilla gan seem to work. at least for MNIST. Should consider making models a little more powerful.
- dropout might kinda help (with the loss anyway)
