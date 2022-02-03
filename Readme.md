# SCFA Lua NN

This project contains a toolset for using machine learning inside __Supreme Commander: Forged Alliance__

As it is based on Lua it might be applicable in another context, but don't bet on it.

# Neural Network Design

## Supported Layouts

  - Perceptron
    - This is currently the only ML algorithm that is supported, if I deem it useful enough others will follow

## Data Layout
The __Neural Network__ is represented as multiple matrices which in turn represent the connections for each neuron to another neuron.

__Example:__

An network with one input node, one hidden layer with 3 node and two output nodes.

                   /-( hidden node 1 )--\---( output node 1 )
                  /                      \ /  /
    ( input ) ---<---( hidden node 2 )----X--<
                  \                      / \  \
                   \-( hidden node 3 )--/---( output node 2 )

All numbers represent the weights for each connection

    Network = {
        {               -- 1x3 Matrix
            {
                0.548,  -- input to hidden node 1
                0.847,  -- input to hidden node 2
                0.22    -- input to hidden node 3
            }
        },
        {               -- 3x2 Matrix
            {
                -0.324, -- hidden node 1 to output node 1
                0.1,    -- hidden node 1 to output node 2
            }
            {
                0.3132  -- hidden node 2 to output node 1
                2.22,   -- hidden node 2 to output node 2
            }   
            {
                -0.387, -- hidden node 3 to output node 1
                0.005   -- hidden node 3 to output node 2
            }
        }
    } 

## Maths

The chosen activation function for neurons is the Sigmoid function

    1 / (1 + math.exp(-x))

