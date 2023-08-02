from synthsurf.linalg import dot
import torch


def normalize_(x):
    x /= dot(x, x).sqrt()
    return x

def normalized(x):
    return x / dot(x, x).sqrt()

def normsq(x):
    return dot(x, x)

def norm(x):
    return normsq(x).sqrt()

def cross(a, b):
    c = torch.zeros_like(a)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c



# def sqdist_unsigned_3dmethod(point, vertices):
#     diff  = point - vertices[0]
#     edge1 = vertices[1] - vertices[0]
#     edge2 = vertices[2] - vertices[0]
#     normal = cross(edge1, edge2)
#     cosalpha = dot(diff, normal) / (norm(diff) * norm(normal))
#     diffproj = norm(diff) * cosalpha
#     pointproj = point - diffproj * normalized(normal)

#     v0 = normalized(edge1) + normalized(edge2)
#     v1 = normalized(vertices[2] - vertices[1]) + normalized(edge1)
#     v2 = normalized(vertices[2] - vertices[1]) + normalized(edge2)

#     f0 = dot(cross(v0, pointproj - vertices[0]))
#     f1 = dot(cross(v1, pointproj - vertices[1]))
#     f2 = dot(cross(v2, pointproj - vertices[2]))

#     inside_triangle = 



def sqdist_unsigned(point, vertices):
    v0, v1, v2 = vertices
    diff = v0 - point
    edge0 = v1 - v0
    edge1 = v2 - v0
    a00 = dot(edge0, edge0)
    a01 = dot(edge0, edge1)
    a11 = dot(edge1, edge1)
    b0 = dot(diff, edge0)
    b1 = dot(diff, edge1)
    c = dot(diff, diff)
    det = abs(a00 * a11 - a01 * a01)
    s = a01 * b1 - a11 * b0
    t = a01 * b0 - a00 * b1

    d2    = -1

    if (s + t <= det):
    
        if (s < 0):
        
            if (t < 0):
            
                if (b0 < 0):
                
                    t = 0
                    if (-b0 >= a00):
                    
                        nearest_entity = 'V1'
                        s = 1
                        d2 = a00 + 2 * b0 + c
                    
                    else:
                    
                        nearest_entity = 'E01'
                        s = -b0 / a00
                        d2 = b0 * s + c
                    
                
                else:
                
                    s = 0
                    if (b1 >= 0):
                    
                        nearest_entity = 'V0'
                        t = 0
                        d2 = c
                    
                    elif (-b1 >= a11):
                    
                        nearest_entity = 'V2'
                        t = 1
                        d2 = a11 + 2 * b1 + c

                    
                    else:
                    
                        nearest_entity = 'E02'
                        t = -b1 / a11
                        d2 = b1 * t + c
                    
                
            
            else:
            
                s = 0
                if (b1 >= 0):
                
                    nearest_entity = 'V0'
                    t = 0
                    d2 = c
                
                elif (-b1 >= a11):
                
                    nearest_entity = 'V2'
                    t = 1
                    d2 = a11 + 2 * b1 + c
                
                else:
                
                    nearest_entity = 'E02'
                    t = -b1 / a11
                    d2 = b1 * t + c
                
            
        
        elif (t < 0):
        
            t = 0
            if (b0 >= 0):
            
                nearest_entity = 'V0'
                s = 0
                d2 = c
            
            elif (-b0 >= a00):
            
                nearest_entity = 'V1'
                s = 1
                d2 = a00 + 2 * b0 + c
            
            else:
            
                nearest_entity = 'E01'
                s = -b0 / a00
                d2 = b0 * s + c
            
        
        else:
        
            nearest_entity = 'F'
            # minimum at interior point
            invDet = 1 / det
            s *= invDet
            t *= invDet
            d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c

        
    
    else:
    
        if (s < 0):
        
            tmp0 = a01 + b0
            tmp1 = a11 + b1
            if (tmp1 > tmp0):
            
                numer = tmp1 - tmp0
                denom = a00 - 2 * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V1'
                    s = 1
                    t = 0
                    d2 = a00 + 2 * b0 + c
                
                else:
                
                    nearest_entity = 'E12'
                    s = numer / denom
                    t = 1 - s
                    d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c
                
            
            else:
            
                s = 0
                if (tmp1 <= 0):
                
                    nearest_entity = 'V2'
                    t = 1
                    d2 = a11 + 2 * b1 + c
                
                elif (b1 >= 0):
                
                    nearest_entity = 'V0'
                    t = 0
                    d2 = c
                
                else:
                
                    nearest_entity = 'E02'
                    t = -b1 / a11
                    d2 = b1 * t + c
                
            
        
        elif (t < 0):
        
            tmp0 = a01 + b1
            tmp1 = a00 + b0
            if (tmp1 > tmp0):
            
                numer = tmp1 - tmp0
                denom = a00 - 2 * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V2'
                    t = 1
                    s = 0
                    d2 = a11 + 2 * b1 + c
                
                else:
                
                    nearest_entity = 'E12'
                    t = numer / denom
                    s = 1 - t
                    d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c
                
            
            else:
            
                t = 0
                if (tmp1 <= 0):
                
                    nearest_entity = 'V1'
                    s = 1
                    d2 = a00 + 2 * b0 + c
                
                elif (b0 >= 0):
                
                    nearest_entity = 'V0'
                    s = 0
                    d2 = c
                
                else:
                
                    nearest_entity = 'E01'
                    s = -b0 / a00
                    d2 = b0 * s + c
                
            
        
        else:
        
            numer = a11 + b1 - a01 - b0
            if (numer <= 0):
            
                nearest_entity = 'V2'
                s = 0
                t = 1
                d2 = a11 + 2 * b1 + c
            
            else:
            
                denom = a00 - 2 * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V1'
                    s = 1
                    t = 0
                    d2 = a00 + 2 * b0 + c
                
                else:
                
                    nearest_entity = 'E12'
                    s = numer / denom
                    t = 1 - s
                    d2 = s * (a00 * s + a01 * t + 2 * b0) + t * (a01 * s + a11 * t + 2 * b1) + c
                
            
    nearest_point = vertices[0] + s * edge0 + t * edge1

    if (d2 < 0):
        print("neg dist: {} true dist: {} entity = {}".format(d2, dot(nearest_point - point, nearest_point - point), nearest_entity))
        d2 = 0

    return d2, nearest_point, nearest_entity



def origfn(point, v0, v1, v2):

    diff = v0 - point
    edge0 = v1 - v0
    edge1 = v2 - v0
    a00 = dot(edge0, edge0)
    a01 = dot(edge0, edge1)
    a11 = dot(edge1, edge1)
    b0 = dot(diff, edge0)
    b1 = dot(diff, edge1)
    c = dot(diff, diff)
    det = abs(a00 * a11 - a01 * a01)
    s = a01 * b1 - a11 * b0
    t = a01 * b0 - a00 * b1

    d2 = -1.0

    if (s + t <= det):
    
        if (s < 0):
        
            if (t < 0):
            
                if (b0 < 0):
                
                    t = 0
                    if (-b0 >= a00):
                    
                        nearest_entity = 'V1'
                        s = 1
                        d2 = a00 + (2) * b0 + c
                    
                    else:
                    
                        nearest_entity = 'E01'
                        s = -b0 / a00
                        d2 = b0 * s + c
                    
                
                else:
                
                    s = 0
                    if (b1 >= 0):
                    
                        nearest_entity = 'V0'
                        t = 0
                        d2 = c
                    
                    elif (-b1 >= a11):
                    
                        nearest_entity = 'V2'
                        t = 1
                        d2 = a11 + (2) * b1 + c
                    
                    else:
                    
                        nearest_entity = 'E02'
                        t = -b1 / a11
                        d2 = b1 * t + c
                    
                
            
            else:
            
                s = 0
                if (b1 >= 0):
                
                    nearest_entity = 'V0'
                    t = 0
                    d2 = c
                
                elif (-b1 >= a11):
                
                    nearest_entity = 'V2'
                    t = 1
                    d2 = a11 + (2) * b1 + c
                
                else:
                
                    nearest_entity = 'E02'
                    t = -b1 / a11
                    d2 = b1 * t + c
                
            
        
        elif (t < 0):
        
            t = 0
            if (b0 >= 0):
            
                nearest_entity = 'V0'
                s = 0
                d2 = c
            
            elif (-b0 >= a00):
            
                nearest_entity = 'V1'
                s = 1
                d2 = a00 + (2) * b0 + c
            
            else:
            
                nearest_entity = 'E01'
                s = -b0 / a00
                d2 = b0 * s + c
            
        
        else:
        
            nearest_entity = 'F'
            invDet = (1) / det
            s *= invDet
            t *= invDet
            d2 = s * (a00 * s + a01 * t + (2) * b0) + t * (a01 * s + a11 * t + (2) * b1) + c
        
    
    else:
    
        tmp0, tmp1, numer, denom

        if (s < 0):
        
            tmp0 = a01 + b0
            tmp1 = a11 + b1
            if (tmp1 > tmp0):
            
                numer = tmp1 - tmp0
                denom = a00 - (2) * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V1'
                    s = 1
                    t = 0
                    d2 = a00 + (2) * b0 + c
                
                else:
                
                    nearest_entity = 'E12'
                    s = numer / denom
                    t = 1 - s
                    d2 = s * (a00 * s + a01 * t + (2) * b0) + t * (a01 * s + a11 * t + (2) * b1) + c
                
            
            else:
            
                s = 0
                if (tmp1 <= 0):
                
                    nearest_entity = 'V2'
                    t = 1
                    d2 = a11 + (2) * b1 + c
                
                elif (b1 >= 0):
                
                    nearest_entity = 'V0'
                    t = 0
                    d2 = c
                
                else:
                
                    nearest_entity = 'E02'
                    t = -b1 / a11
                    d2 = b1 * t + c
                
            
        
        elif (t < 0):
        
            tmp0 = a01 + b1
            tmp1 = a00 + b0
            if (tmp1 > tmp0):
            
                numer = tmp1 - tmp0
                denom = a00 - (2) * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V2'
                    t = 1
                    s = 0
                    d2 = a11 + (2) * b1 + c
                
                else:
                
                    nearest_entity = 'E12'
                    t = numer / denom
                    s = 1 - t
                    d2 = s * (a00 * s + a01 * t + (2) * b0) + t * (a01 * s + a11 * t + (2) * b1) + c
                
            
            else:
            
                t = 0
                if (tmp1 <= 0):
                
                    nearest_entity = 'V1'
                    s = 1
                    d2 = a00 + (2) * b0 + c
                
                elif (b0 >= 0):
                
                    nearest_entity = 'V0'
                    s = 0
                    d2 = c
                
                else:
                
                    nearest_entity = 'E01'
                    s = -b0 / a00
                    d2 = b0 * s + c
                
            
        
        else:
        
            numer = a11 + b1 - a01 - b0
            if (numer <= 0):
            
                nearest_entity = 'V2'
                s = 0
                t = 1
                d2 = a11 + (2) * b1 + c
            
            else:
            
                denom = a00 - (2) * a01 + a11
                if (numer >= denom):
                
                    nearest_entity = 'V1'
                    s = 1
                    t = 0
                    d2 = a00 + (2) * b0 + c
                
                else:
                
                    nearest_entity = 'E12'
                    s = numer / denom
                    t = 1 - s
                    d2 = s * (a00 * s + a01 * t + (2) * b0) + t * (a01 * s + a11 * t + (2) * b1) + c
                
            
        
    if (d2 < 0):
        print("boo")
        d2 = 0
    

    nearest_point = v0 + s * edge0 + t * edge1
    return d2, nearest_point, nearest_entity



point = torch.as_tensor([-30.456694, -17.000000, 23.913385])
vertices = torch.as_tensor([
    [-11.593674, -75.367928, -20.127497],
    [-7.548377, -73.445557, -20.802279],
    [-9.209785, -76.319008, -21.476013]
])

d2, nearest_point, nearest_entity = sqdist_unsigned(point, vertices)
d2, nearest_point, nearest_entity = origfn(point, *vertices)

# nearest = [5842.029628 2705.712212 -996.836279]

foo = 0