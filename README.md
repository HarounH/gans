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


## InfoGAN notes
- c|x requires softmax, not sigmoid. its a probability distrib
- c|x can be coupled with D(x), but need not be
- do infogans train slower (dL/#epochs)?
- Why is zdim so much smaller?
- WHY IS LEAKYRELU so BAD?!

## WGAN
- smaller lr
- weight clipping on critic needed for guarantees.
- https://arxiv.org/pdf/1704.00028.pdf says we can use a gradient norm in loss instead... a little harder to implement, isn't it?

## MRGAN
- regularize training by using encoder.
- having generator being able to generate encoded x makes it get modes. yay.
- what exactly is the MDGAN?
