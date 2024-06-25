class Parameters():
  # Registration algorithm ----------------------------------------------------------------
  class Registration():
    VOXEL_SIZE = 30
    MAX_NN_NORMALS = 30
    MAX_NN_FPFH = 100
    CONFIDENCE = 1.0
    MAX_ITERATION_RANSAC = 1000000
    EPSILON = 1e-6
    MAX_ITERATION_ICP = 50
    RANSAC_LOOP_SIZE = 5
  
  # Bucket point removal algorithm --------------------------------------------------------
  class BucketRemoval():
    THRESHOLD_DISTANCE = 20
    NB_NEIGHBORS = 40
    STD_RATIO = 1
    NB_POINTS = 20
    RADIUS = 100
    
  # Load and bucket points merge algorithm ----------------------------------------------
  class MergePoints():
    DISTANCE_THRESHOLD = 120 
    DETECTION_THRESHOLD = 20
    ANGULAR_STEP = 25
    SLOPE = 500
    NB_NEIGHBORS = 10
    STD_RATIO = 20
    
  class MeshReconstruction():
    ALPHA = 150
    N_FILTER_ITERATIONS = 5
