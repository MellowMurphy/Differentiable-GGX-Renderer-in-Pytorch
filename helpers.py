# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:59:52 2020

@author: Christian Murphy
"""

import os
import torch
import numpy as np
import math

def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1

def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2

#Log a tensor and normalize it.
def logTensor(tensor):
    return  (torch.log(torch.add(tensor,0.01)) - torch.log(0.01)) / (torch.log(1.01)-torch.log(0.01))

def rand_range(shape, low, high, dtype=torch.float32):
    return ((high - low) * torch.rand(shape,dtype=dtype))

def randn_range(shape, mean, std, dtype=torch.float32):
    return (torch.randn(shape,dtype=dtype) * std + mean)

#Generate a random direction on the upper hemisphere with gaps on the top and bottom of Hemisphere. Equation is described in the Global Illumination Compendium (19a)
def generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.05):
    r1 = rand_range((batchSize, nbRenderings, 1), 0.0 + lowEps, 1.0 - highEps, dtype=torch.float32)
    r2 =  torch.rand([batchSize, nbRenderings, 1], dtype=torch.float32)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
    #min alpha = atan(sqrt(1-r^2)/r)
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    finalVec = torch.cat([x, y, z], axis=-1) #Dimension here should be [batchSize,nbRenderings, 3]
    return finalVec

#Remove the gamma of a vector
def removeGamma(tensor):
    return torch.pow(tensor, 2.2)

#Add gamma to a vector
def addGamma(tensor):
    return torch.pow(tensor, 0.4545)

# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def normalize(tensor):
    Length = torch.sqrt(torch.sum(torch.square(tensor), dim = -1, keepdim=True))
    return torch.div(tensor, Length)

#Generate a distance to compute for the specular renderings (as position is important for this kind of renderings)
def generate_distance(batchSize, nbRenderings):
    gaussian = randn_range([batchSize, nbRenderings, 1], 0.5, 0.75, dtype=torch.float32) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (torch.exp(gaussian))

#Very small lamp attenuation
def lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*torch.square(distance))

#Physically based lamp attenuation
def lampAttenuation_pbr(distance):
    return 1.0 / torch.square(distance)

#Clip values between min an max
def squeezeValues(tensor, min, max):
    return torch.clamp(tensor, min, max)

def DotProduct(tensorA, tensorB):
    return torch.sum(torch.mul(tensorA, tensorB), dim = -1, keepdim=True)

# Generate an array grid between -1;1 to act as the "coordconv" input layer (see coordconv paper)
def generateCoords(inputShape):
    crop_size = inputShape[-2]
    firstDim = inputShape[0]

    Xcoords= torch.unsqueeze(torch.linspace(-1.0, 1.0, crop_size), axis=0)
    Xcoords = Xcoords.repeat(crop_size, 1)
    Ycoords = -1 * Xcoords.T #put -1 in the bottom of the table
    Xcoords = torch.unsqueeze(Xcoords, axis = -1)
    Ycoords = torch.unsqueeze(Ycoords, axis = -1)
    coords = torch.cat([Xcoords, Ycoords], axis=-1)
    coords = torch.unsqueeze(coords, axis = 0)#Add dimension to support batch size and nbRenderings should now be [1, 256, 256, 2].
    coords = coords.repeat(firstDim, 1, 1, 1) #Add the proper dimension here for concat
    return coords

# Generate an array grid between -1;1 to act as each pixel position for the rendering.
def generateSurfaceArray(crop_size, pixelsToAdd = 0):
    totalSize = crop_size + (pixelsToAdd * 2)
    surfaceArray=[]
    XsurfaceArray = torch.unsqueeze(torch.linspace(-1.0, 1.0, totalSize), axis=0)
    XsurfaceArray = XsurfaceArray.repeat(totalSize, 1)
    YsurfaceArray = -1 * XsurfaceArray.T #put -1 in the bottom of the table
    XsurfaceArray = torch.unsqueeze(XsurfaceArray, axis = -1)
    YsurfaceArray = torch.unsqueeze(YsurfaceArray, axis = -1)

    surfaceArray = torch.cat([XsurfaceArray, YsurfaceArray, torch.zeros([totalSize, totalSize,1], dtype=torch.float32)], axis=-1)
    surfaceArray = torch.unsqueeze(torch.unsqueeze(surfaceArray, axis = 0), axis = 0)#Add dimension to support batch size and nbRenderings
    return surfaceArray

#create small variation to be added to the positions of lights or camera.
def jitterPosAround(batchSize, nbRenderings, posTensor, mean = 0.0, stddev = 0.03):
    randomPerturbation =  torch.clamp(randn_range([batchSize, nbRenderings,1,1,1,3], mean, stddev, dtype=torch.float32), -0.24, 0.24) #Clip here how far it can go to 8 * stddev to avoid negative values on view or light ( Z minimum value is 0.3)
    return posTensor + randomPerturbation

#Adds a little bit of noise
def addNoise(renderings):
    shape = renderings.shape
    stddevNoise = torch.exp(randn_range(1, mean = torch.log(0.005), stddev=0.3))
    noise = randn_range(shape, mean=0.0, std=stddevNoise)
    return renderings + noise

#generate the diffuse rendering for the loss computation
def generateDiffuseRendering(batchSize, nbRenderings, targets, outputs, renderer):
    currentViewPos = generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightPos = generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)

    wi = currentLightPos
    wi = torch.unsqueeze(wi, axis=2)
    wi = torch.unsqueeze(wi, axis=2)

    wo = currentViewPos
    wo = torch.unsqueeze(wo, axis=2)
    wo = torch.unsqueeze(wo, axis=2)

    #Add a dimension to compensate for the nb of renderings
    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)

    #Here we have wi and wo with shape [batchSize, height,width, nbRenderings, 3]
    renderedDiffuse = renderer.render(targets,wi,wo, None, "diffuse", useAugmentation = False, lossRendering = True)[0]

    renderedDiffuseOutputs = renderer.render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]#tf_Render_Optis(outputs,wi,wo)
    #renderedDiffuse = tf.Print(renderedDiffuse, [tf.shape(renderedDiffuse)],  message="This is renderings targets Diffuse: ", summarize=20)
    #renderedDiffuseOutputs = tf.Print(renderedDiffuseOutputs, [tf.shape(renderedDiffuseOutputs)],  message="This is renderings outputs Diffuse: ", summarize=20)
    return [renderedDiffuse, renderedDiffuseOutputs]

#generate the specular rendering for the loss computation
def generateSpecularRendering(batchSize, nbRenderings, surfaceArray, targets, outputs, renderer):
    currentViewDir = generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightDir = currentViewDir * (torch.tensor((-1.0, -1.0, 1.0)).unsqueeze(0))
    #Shift position to have highlight elsewhere than in the center.
    currentShift = torch.cat([rand_range([batchSize, nbRenderings, 2], -1.0, 1.0), torch.zeros([batchSize, nbRenderings, 1], dtype=torch.float32) + 0.0001], axis=-1)

    currentViewPos = torch.mul(currentViewDir, generate_distance(batchSize, nbRenderings)) + currentShift
    currentLightPos = torch.mul(currentLightDir, generate_distance(batchSize, nbRenderings)) + currentShift

    currentViewPos = torch.unsqueeze(currentViewPos, axis=2)
    currentViewPos = torch.unsqueeze(currentViewPos, axis=2)

    currentLightPos = torch.unsqueeze(currentLightPos, axis=2)
    currentLightPos = torch.unsqueeze(currentLightPos, axis=2)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)
    #targets = tf.Print(targets, [tf.shape(targets)],  message="This is targets in specu renderings: ", summarize=20)
    renderedSpecular = renderer.render(targets,wi,wo, None, "specu", useAugmentation = False, lossRendering = True)[0]
    renderedSpecularOutputs = renderer.render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]
    #tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)

    #renderedSpecularOutputs = tf.Print(renderedSpecularOutputs, [tf.shape(renderedSpecularOutputs)],  message="This is renderings outputs Specular: ", summarize=20)
    return [renderedSpecular, renderedSpecularOutputs]

