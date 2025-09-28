# Observation and Eureka Moments 

## Observations
- AF must have few conditions like 
    + Non-Linearity
    + Smooth Derivative
    + Many to One Mapping
    + Computationally Efficient
    + Differentiable
    + Range of output values(Except RELU --> upper limit unbounded)

    + AF can be `one to Many` as then the `deterministic nature` of the model is lost also when the derivative is calculated it will be difficult to find the `correct direction of gradient descent`

RELU, Sigmoid, tanh these AF are the most used ones in deep learning these activation functions are non-linear in nature also have smooth derivatives  
they have the tendencies of `Many to One` mapping
RELU is the most used one in `hidden layers`
Sigmoid and Tanh are used in the output layer for `binary classification and multi-class classification`


