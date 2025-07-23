"""
How many perceptions are used in the input layer?

  The image is 10x10 pixels so 100 pixels
  I'm going to treat each pixel as its own perception detailing --> 100 perceptions

How many perceptions are used in the output layer?

  Output values are classified from 0 to 9 which allow for 10 possible classifications
  So, the output layer has 10 perceptions

How many perceptions are used in the fully connected layers?

  The network has 3 layers with 10 perceptions each
  3 * 10 = 30 perceptions

"""
