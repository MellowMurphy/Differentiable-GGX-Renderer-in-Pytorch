# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:11:55 2020

@author: smcmurphy
"""

import render_loss
import helpers
import torch
import skimage.io as io

def main():
    batchSize = 2
    nbRenderings = 9
    imgSize = 256
    DEPTH = 12
    
    #Using random noise for test purposes, fill with texture maps instead
    #Order is normals, diffuse, roughness, specular with 3 channels each
    targets = torch.rand(batchSize,imgSize,imgSize,DEPTH)
    outputs = torch.rand(batchSize,imgSize,imgSize,DEPTH)
    
    renderer = render_loss.GGXRenderer()
    
    surface = helpers.generateSurfaceArray(imgSize)
    diffuse_result = helpers.generateDiffuseRendering(batchSize, nbRenderings, targets, outputs, renderer)
    specular_result = helpers.generateSpecularRendering(batchSize, nbRenderings, surface, targets, outputs, renderer)
     
    first_set_diffuse = diffuse_result[0][0]
    first_set_diffuse = first_set_diffuse.reshape(nbRenderings*imgSize,imgSize,3)
    io.imsave("first_set_diffuse.jpg",first_set_diffuse)
    
    first_set_specular = specular_result[0][0]
    first_set_specular = first_set_specular.reshape(nbRenderings*imgSize,imgSize,3)
    io.imsave("first_set_specular.jpg",first_set_specular)
    

    
main()