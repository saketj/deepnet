Authors: Saket Saurabh, Shashank Gupta

Course Project: ME 759

Project Title: Performance Evaluation of Image Recognition 
               using Neural Network Algorithms over GPU

Pre-requisities: >= GCC 4.8, >= CUDA 7.0, >= CMAKE 2.8


Folder structure:
    The src/ folder contains the entire source code, while the dataset is contained
    inside the mnist/ folder. The build/ folder is empty and will be used to build
    the binaries for the project.


To Build:
    Run the following commands from the root directory to build the project:
          cd build
          cmake ../src
          make
    This will create two binaries 'deepnet' and 'deepnet-cuda' inside the bin
    directory.


To Execute:
    #########################         IMPORTANT    ##################################
          To run the binaries, you need to be inside the 'build/' folder, because the
          binaries search for the dataset paths relative to the build directory.
    #########################         IMPORTANT    ##################################
    Run the following commands from the root directory to run:
          cd build
          bin/deepnet         # runs the sequential version
          bin/deepnet-cuda    # runs the parallel version
