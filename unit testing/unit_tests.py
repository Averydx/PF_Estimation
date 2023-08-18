import numpy as np

def resample_test(): 
    test_particles = np.array([[10,1,0],[10,5,1],[10,1,1],[8,2,3],[7,6,2],[6,6,2],[6,7,4],[10,2,3],[10,4,4],[1,1,1]])
    test_indices = np.array([1,1,1,4,2,1,3,0,4,3])

    valid = np.array([[10,5,1],[10,5,1],[10,5,1],[7,6,2],[10,1,1],[10,5,1],[8,2,3],[10,1,0],[7,6,2],[8,2,3]])

    if(len(test_indices) == len(test_particles)): 
       
       particle_copy = np.copy(test_particles)

       for i,_ in enumerate(test_particles):
            test_particles[i] = particle_copy[int(test_indices[i])]

    if np.all(np.equal(test_particles,valid)): 
        print(f"Resampling...Succeeded")
    else : 
        print("Resampling...Failed")
       


