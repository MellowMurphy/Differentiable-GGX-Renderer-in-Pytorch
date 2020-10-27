# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:47:34 2020

@author: Christian Murphy
"""

import torch
import helpers #dont have this yet
import math
import numpy as np

class GGXRenderer:
    includeDiffuse = True
    
    def __init__(self, includeDiffuse = True):
        self.includeDiffuse = includeDiffuse
        
    #Compute the diffuse part of the equation
    def compute_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi
    
    #Compute the distribution function D driving the statistical orientation of the micro facets.
    def compute_distribution(self, roughness, NdotH):
        alpha = torch.square(roughness)
        underD = 1/torch.clamp((torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0), min = 0.001)
        return (torch.square(alpha * underD)/math.pi)
    
    #Compute the fresnel approximation F
    def compute_fresnel(self, specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH);
        return specular + (1.0 - specular) * sphg
    
    #Compute the Geometry term (also called shadowing and masking term) G taking into account how microfacets can shadow each other.
    def compute_geometry(self, roughness, NdotL, NdotV):
        return self.G1(NdotL, torch.square(roughness)/2) * self.G1(NdotV, torch.square(roughness)/2)

    def G1(self, NdotW, k):
        return 1.0/torch.clamp((NdotW * (1.0 - k) + k), min = 0.001)
    
    #This computes the equations of Cook-Torrance for a BRDF without taking light power etc... into account.
    def calculateBRDF(self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight):

        h = helpers.normalize(torch.add(wiNorm, woNorm) / 2.0)
        #Put all the parameter values between 0 and 1 except the normal as they should be used between -1 and 1 (as they express a direction in a 360° sphere)        
        normals = torch.unsqueeze(svbrdf[:,:,:,0:3], axis=1)
        normals = helpers.normalize(normals)
        diffuse = torch.unsqueeze(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,3:6]), 0.0,1.0), dim = 1)
        roughness = torch.unsqueeze(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,6:9]), 0.0, 1.0), dim = 1)
        specular = torch.unsqueeze(helpers.squeezeValues(helpers.deprocess(svbrdf[:,:,:,9:12]), 0.0, 1.0), dim = 1)
        #Avoid roughness = 0 to avoid division by 0
        
        roughness = torch.clamp(roughness, min = 0.001)

        #If we have multiple lights to render, add a dimension to handle it.
        if multiLight:
            diffuse = torch.unsqueeze(diffuse, dim = 1)
            normals = torch.unsqueeze(normals, dim = 1)
            specular = torch.unsqueeze(specular, dim = 1)
            roughness = torch.unsqueeze(roughness, dim = 1)

        NdotH = helpers.DotProduct(normals, h)
        NdotH[NdotH != NdotH] = 0
        
        NdotL = helpers.DotProduct(normals, wiNorm)
        NdotL[NdotL != NdotL] = 0
        
        NdotV = helpers.DotProduct(normals, woNorm)
        NdotV[NdotV != NdotV] = 0

        VdotH = helpers.DotProduct(woNorm, h)

        diffuse_rendered = self.compute_diffuse(diffuse, specular)
        D_rendered = self.compute_distribution(roughness, torch.clamp(NdotH, min = 0.0))
        G_rendered = self.compute_geometry(roughness, torch.clamp(NdotL,min = 0.0), torch.clamp(NdotV, min = 0.0))
        F_rendered = self.compute_fresnel(specular, torch.clamp(VdotH, min = 0.0))
        
        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered
        
        #Add the diffuse part of the rendering if required.        
        if self.includeDiffuse:
            result = result + diffuse_rendered
        return result, NdotL
    
    def render(self, svbrdf, wi, wo, currentConeTargetPos, tensorboard = "", multiLight = False, currentLightPos = None, lossRendering = True, isAmbient = False, useAugmentation = True):
        wiNorm = helpers.normalize(wi)
        woNorm = helpers.normalize(wo)

        #Calculate how the image should look like with completely neutral lighting
        result, NdotL = self.calculateBRDF(svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight)
        resultShape = result.shape
        lampIntensity = 1.5
        
        #midway = result[0][0]
        #io.imsave("midway.jpg",midway)
        
        result = torch.from_numpy(np.load("result.npy"))
        NdotL =  torch.from_numpy(np.load("NdotL.npy")) #np.save("NdotL.npy",NdotL.numpy())
        
        #Add lighting effects
        if not currentConeTargetPos is None:
            #If we want a cone light (to have a flash fall off effect)
            currentConeTargetDir = currentLightPos - currentConeTargetPos #currentLightPos should never be None when currentConeTargetPos isn't
            coneTargetNorm = helpers.normalize(currentConeTargetDir)
            distanceToConeCenter = (torch.maximum(0.0, helpers.DotProduct(wiNorm, coneTargetNorm)))
        if not lossRendering:
            #If we are not rendering for the loss
            if not isAmbient:
                if useAugmentation:
                    #The augmentations will allow different light power and exposures                 
                    stdDevWholeBatch = torch.exp(torch.randn((), mean = -2.0, stddev = 0.5))
                    #add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                    lampIntensity = torch.abs(torch.randn((resultShape[0], resultShape[1], 1, 1, 1), mean = 10.0, stddev = stdDevWholeBatch)) # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                    #autoExposure
                    autoExposure = torch.exp(torch.randn((), mean = np.log(1.5), stddev = 0.4))
                    lampIntensity = lampIntensity * autoExposure
                else:
                    lampIntensity = torch.reshape(torch.FloatTensor(13.0), [1, 1, 1, 1, 1]) #Look at the renderings when not using augmentations
            else:
                #If this uses ambient lighting we use much small light values to not burn everything.
                if useAugmentation:
                    lampIntensity = torch.exp(torch.randn((resultShape[0], 1, 1, 1, 1), mean = torch.log(0.15), stddev = 0.5)) #No need to make it change for each rendering.
                else:
                    lampIntensity = torch.reshape(torch.FloatTensor(0.15), [1, 1, 1, 1, 1])
            #Handle light white balance if we want to vary it..
            if useAugmentation and not isAmbient:
                whiteBalance = torch.abs(torch.randn([resultShape[0], resultShape[1], 1, 1, 3], mean = 1.0, stddev = 0.03))
                lampIntensity = lampIntensity * whiteBalance

            if multiLight:
                lampIntensity = torch.unsqueeze (lampIntensity, axis = 2) #add a constant dim if using multiLight
        lampFactor = lampIntensity * math.pi

        if not isAmbient:
            if not lossRendering:
                #Take into accound the light distance (and the quadratic reduction of power)            
                lampDistance = torch.sqrt(torch.sum(torch.square(wi), axis = -1, keep_dims=True))

                lampFactor = lampFactor * helpers.lampAttenuation_pbr(lampDistance)
            if not currentConeTargetPos is None:
                #Change the exponent randomly to simulate multiple flash fall off.            
                if useAugmentation:
                    exponent = torch.exp(torch.randn((), mean=np.log(5), stddev=0.35))
                else:
                    exponent = 5.0
                lampFactor = lampFactor * torch.pow(distanceToConeCenter, exponent)
                print("using the distance to cone center")

        
        result = result * lampFactor
        result = result * torch.clamp(NdotL, min = 0.0)
        
        if multiLight:
            result = torch.sum(result, axis = 2) * 1.0#if we have multiple light we need to multiply this by (1/number of lights).
        if lossRendering:
            result = result / torch.unsqueeze(torch.clamp(wiNorm[:,:,:,:,2], min = 0.001), axis=-1)# This division is to compensate for the cosinus distribution of the intensity in the rendering.
        
        return [result]#, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]
    
