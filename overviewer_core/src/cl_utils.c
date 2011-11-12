

#include "overviewer.h"

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#else
#include <CL/OpenCl.h>
#endif

#include <sys/stat.h>

static int static_var = 0;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;


static const char* getClError(cl_int e) {
    if (e == CL_SUCCESS)
        return "CL_SUCCESS";
    if (e == CL_INVALID_PROGRAM)
        return "CL_INVALID_PROGRAM";
    if (e == CL_INVALID_VALUE)
        return "CL_INVALID_VALUE";
    if (e == CL_INVALID_DEVICE)
        return "CL_INVALID_DEVICE";
    if (e == CL_INVALID_OPERATION)
        return "CL_INVALID_OPERATION";
    if (e == CL_COMPILER_NOT_AVAILABLE)
        return "CL_COMPILER_NOT_AVAILABLE";
    if (e == CL_BUILD_PROGRAM_FAILURE)
        return "CL_BUILD_PROGRAM_FAILURE";
    if (e == CL_OUT_OF_HOST_MEMORY)
        return "CL_OUT_OF_HOST_MEMORY";
    if (e == CL_INVALID_PROGRAM_EXECUTABLE)
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    if (e == CL_INVALID_COMMAND_QUEUE)
        return "CL_INVALID_COMMAND_QUEUE";
    if (e == CL_INVALID_KERNEL)
        return "CL_INVALID_KERNEL";
    if (e == CL_INVALID_CONTEXT)
        return "CL_INVALID_CONTEXT";
    if (e == CL_INVALID_KERNEL_ARGS)
        return "CL_INVALID_KERNEL_ARGS";
    if (e == CL_INVALID_WORK_DIMENSION)
        return "CL_INVALID_WORK_DIMENSION";
    if (e == CL_INVALID_WORK_GROUP_SIZE)
        return "CL_INVALID_WORK_GROUP_SIZE";
    if (e == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    if (e == CL_INVALID_IMAGE_SIZE)
        return "CL_INVALID_IMAGE_SIZE";
    if (e == CL_INVALID_HOST_PTR)
        return "CL_INVALID_HOST_PTR";
    if (e == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    if (e == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    if (e == CL_INVALID_ARG_INDEX)
        return "CL_INVALID_ARG_INDEX";
    if (e == CL_INVALID_ARG_VALUE)
        return "CL_INVALID_ARG_VALUE";
    if (e == CL_INVALID_MEM_OBJECT)
        return "CL_INVALID_MEM_OBJECT";
    if (e == CL_INVALID_SAMPLER) 
        return "CL_INVALID_SAMPLER";
    if (e == CL_INVALID_ARG_SIZE)
        return "CL_INVALID_ARG_SIZE";
    if (e == CL_INVALID_QUEUE_PROPERTIES) 
        return "CL_INVALID_QUEUE_PROPERTIES";
    if (e == CL_INVALID_PLATFORM)
        return "CL_INVALID_PLATFORM";
    if (e == CL_INVALID_DEVICE_TYPE)
        return "CL_INVALID_DEVICE_TYPE";
    if (e == CL_DEVICE_NOT_FOUND)
        return "CL_DEVICE_NOT_FOUND"; 

    return "UNKNOWN";
}



PyObject *
do_cl_init(PyObject *self, PyObject *args)
{
    /* Get the device that we will use */

    cl_device_id    work_device;
    cl_int error;
    const char *program_name = "paste.cl";
    struct stat s;
    FILE* prog_file;
    size_t filesize, read_so_far;
    char * source;
    size_t len;
    cl_uint num_plats;
    cl_platform_id platform;

    if (clGetPlatformIDs(1, &platform, 0) != CL_SUCCESS) {
        printf("Failed to get plats\n");
        return NULL;
    }

    printf("getting device ids\n");
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &work_device, NULL);
    if (error != CL_SUCCESS) {
        printf("Error: %s\n", getClError(error));
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clGetDeviceIDs");
        return NULL;
    }

    printf("creating context\n");
    context =clCreateContext(0, 1, &work_device, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clCreateContext");
        return NULL;
    }

    printf("creating command queue\n");
    queue = clCreateCommandQueue(context, work_device, 0, &error);
    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clCreateCommandQueue");
        return NULL;
    }    


    if (stat(program_name,&s) != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed stat paste.cl");
        return NULL;

    }


    prog_file = fopen(program_name, "r");
    filesize = s.st_size+1;
    source = (char*)malloc(filesize);
    printf("Attempting to read %d bytes of paste.cl\n", filesize);
    memset(source, 0, filesize);

    read_so_far = fread(source, 1, filesize-1, prog_file);
    printf("read so far: %d\n", read_so_far);

    if (!feof(prog_file)) {

        PyErr_SetString(PyExc_RuntimeError,
                "Failed read all of paste.cl");
        return NULL;
    }
    fclose(prog_file);

    len = strlen(source);
    printf("Compiling: %s\n", source);

    program = clCreateProgramWithSource(context, 1, (const char**)&source, &len, &error);
    free(source);

    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed create program");

        return NULL;
    }

    error = clBuildProgram(program, 1, &work_device, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        /* fetch and print build log */
        size_t logSize;
        char *buildLog;
        printf("error: %s\n", getClError(error));

        clGetProgramBuildInfo(program, work_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        buildLog = malloc(logSize);
        clGetProgramBuildInfo(program, work_device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, 0);
        printf("-----\nBuild log:\n-----\n\n%s\n\n", buildLog);
        free(buildLog);

        PyErr_SetString(PyExc_RuntimeError,
                "Failed build program");
        printf("error: %s\n", getClError(error));
        return NULL;
    } 

    kernel = clCreateKernel(program, "paste_gpu", &error);
    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed create kernel");
        printf("error: %s\n", getClError(error));
        return NULL;
    }




    return Py_None;
}


/*
 stich_quad_images
 args: 1: PIL image to paste into
       2: list of images and coords in the form
          [[(x,y), "path"], [(x,y), "path"], ... ]
 *
 */
PyObject *
stitch_quad_images(PyObject* self, PyObject *args)
{
    PyObject* destImg;
    PyObject* imgList;
    PyObject* imgsize;

    PyObject* imgsize0_py, *imgsize1_py;
    Imaging imDest;
    cl_int oErr = 0;
    cl_image_format imgFormat = {CL_RGBA, CL_UNSIGNED_INT8};
    int x;
    cl_mem clImg;
    cl_mem quadCLImg[4] = {0}; // to hold our 4 input images
    
    const size_t offset[3] = {0,0,0};
    const size_t region[3] = {384, 384, 1};

    char *destImage;


    if (!PyArg_ParseTuple(args, "OO",  &destImg, &imgList))
        return NULL;

    imDest = imaging_python_to_c(destImg);

    clImg = clCreateImage2D(context, // context
            CL_MEM_WRITE_ONLY, // mem flags
            &imgFormat, // img format
            imDest->xsize, // image widht
            imDest->ysize, // image height
            0, // image pitch
            NULL, // host ptr
            &oErr);

    if (clImg == NULL || oErr != CL_SUCCESS) {
        printf("Something went wrong creating Image2d\n");
    }
    //printf("Created 2dImage with width=%d and height=%d\n", imDest->xsize, imDest->ysize);
       
    oErr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clImg);
    if (oErr != CL_SUCCESS) {
        printf("Error: %s\n", getClError(oErr));
        PyErr_SetString(PyExc_RuntimeError,  "Failed setting kernel args 0");
        return NULL;

    }


    // load up our 4 input images

    for(x=0; x < 4; x++) {
        const unsigned char quadrant = x;
        Imaging img_img;
        void* img_buf;
        size_t global_ws[2] = {192, 192};

        PyObject *img_py = PySequence_GetItem(imgList, x); // new reference
        if (img_py == Py_None) { continue; }
  

        img_img = imaging_python_to_c(img_py);
        //printf("Recieved a img with size %d,%d and pixelsize:%d\n", img_img->xsize, img_img->ysize, img_img->pixelsize);
        img_buf = *(img_img->image);
        //printf("  image buffer is at %p\n", img_buf);

        // create a 2dImage with this data
        quadCLImg[x] = clCreateImage2D(context, // context
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // mem flags
                &imgFormat, // img format
                img_img->xsize, // image widht
                img_img->ysize, // image height
                0, // image pitch
                img_buf, // host ptr
                &oErr);
        if (quadCLImg[x] != NULL) {
            //printf("Created 2d image with hostdata\n");
        } else {
            printf("Failed to create 2dimage with hostdata: %s\n", getClError(oErr));
            return NULL;
        }

        Py_DECREF(img_py);

        oErr = clSetKernelArg(kernel, 1, sizeof(cl_mem), & quadCLImg[x]);
        if (oErr != CL_SUCCESS) {
            printf("Error: %s\n", getClError(oErr));
            PyErr_SetString(PyExc_RuntimeError,  "Failed setting kernel args 1");
            return NULL;
        }

        oErr = clSetKernelArg(kernel, 2, sizeof(cl_uchar), &quadrant);
        if (oErr != CL_SUCCESS) {
            printf("Error: %s\n", getClError(oErr));
            PyErr_SetString(PyExc_RuntimeError,  "Failed setting kernel args 2");
            return NULL;
        }


        // enqueue this shit!
        oErr = clEnqueueNDRangeKernel(queue, // queue
                kernel, // kernel
                2, // work dim
                NULL, // global_work_offset
                global_ws, // global_work_size
                local_ws, // local_work_size
                0,
                NULL,
                NULL);

        if (oErr != CL_SUCCESS) {
            printf("Error: %s\n", getClError(oErr));
            PyErr_SetString(PyExc_RuntimeError,  "Failed to enqueue kernel");
            return NULL;
        }
        //printf("Enqueued kernel\n");


    }



    destImage = *(imDest->image);

    oErr = clEnqueueReadImage(queue, // queue
            clImg, // image to read from
            0, // blocking?
            offset, // offset
            region, // region
            384*4, // row pitch
            0, // slice pitch.  0 if 2d
            destImage, // host ptr
            0, NULL, NULL);
    if (oErr != CL_SUCCESS) {
        printf("Error: %s\n", getClError(oErr));
        PyErr_SetString(PyExc_RuntimeError,  "Failed to enqueue readimage");
        return NULL;
    }


    oErr = clFinish(queue);
    if (oErr != CL_SUCCESS) {
        printf("Error: %s\n", getClError(oErr));
        PyErr_SetString(PyExc_RuntimeError,  "failed to finish queue");
        return NULL;
    }
    //printf("finished queue\n");
    //printf("size of imDest: %d,%d  x%d\n", imDest->xsize, imDest->ysize, imDest->pixelsize);
    //memcpy(imDest->image, dummyBuffer, 10*10*4);
    //printf("memcpy\n");

    // clean up
    if (clImg != NULL)
        clReleaseMemObject(clImg);
    for(x=0; x < 4; x++) {
        if (quadCLImg[x] != NULL)
            clReleaseMemObject(quadCLImg[x]);
    }




    //printf("about to buildValue\n");

    return Py_BuildValue("i", 0);

}


PyObject *
print_cl_info(PyObject *self, PyObject *args)
{

    int i;
    cl_uint num_devs, num_plats;
    cl_int rc;
    cl_platform_id *plats;
    cl_device_id *ids;

    clGetPlatformIDs(0, NULL, &num_plats);
    printf("Total number of platforms: %d\n", num_plats);


    plats = malloc(sizeof(cl_platform_id) *num_plats);
    clGetPlatformIDs(num_plats, plats, 0);
    for (i = 0; i < num_plats; i++) {
        cl_platform_id plat = plats[i];
        char name[120];
        
        printf("Platform #%d\n", i);
        printf("-----------\n");

        if (clGetPlatformInfo(plat, CL_PLATFORM_NAME, 120, name, NULL) == CL_SUCCESS) {
            printf("  Name: %s\n", name);

        }
        if (clGetPlatformInfo(plat, CL_PLATFORM_VENDOR, 120, name, NULL) == CL_SUCCESS) {
            printf("  Vendor: %s\n", name);

        }
        if (clGetPlatformInfo(plat, CL_PLATFORM_VERSION, 120, name, NULL) == CL_SUCCESS) {
            printf("  Version: %s\n", name);
        }
        if (clGetPlatformInfo(plat, CL_PLATFORM_EXTENSIONS, 120, name, NULL) == CL_SUCCESS) {
            printf("  Extensions: %s\n", name);
        }
    }
    printf("\n");

    rc = clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs);
    if (rc != CL_SUCCESS) {
        printf("Failed to clGetDeviceIDs\n");
        return NULL;
    }
    printf("Total number of devices: %d\n", num_devs);

    ids = malloc(sizeof(cl_device_id)*num_devs);
    rc = clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_ALL, num_devs, ids, NULL);
    if (rc != CL_SUCCESS) {
        printf("Failed to clGetDeviceIDs (2) \n");
        return NULL;
    }

    for (i = 0; i < num_devs; i++ ){
        cl_device_id dev = ids[i];
        char name[120];
        cl_device_type type;
        cl_uint uint_ret;
        cl_bool bool_ret;
        size_t actual_size;
        printf("Device #%d\n", i);
        printf("---------\n");

        if (clGetDeviceInfo(dev, CL_DEVICE_NAME, 120, name, &actual_size) == CL_SUCCESS)
            printf("  Name: %s\n", name);
        if (clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 120, name, &actual_size) == CL_SUCCESS)
            printf("  Vendor: %s\n", name);
        if (clGetDeviceInfo(dev, CL_DEVICE_VERSION, 120, name, &actual_size) == CL_SUCCESS)
            printf("  Version: %s\n", name);

        if (clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL) == CL_SUCCESS) {
            printf("  Type: ");
            if (type == CL_DEVICE_TYPE_CPU)
                printf("CPU");
            else if (type == CL_DEVICE_TYPE_GPU)
                printf("GPU");
            else
                printf("Unknown");
            printf("\n");
        }
        if (clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &uint_ret, NULL) == CL_SUCCESS) {
            printf("  Cores: %d\n", uint_ret);
        }

        if (clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &uint_ret, NULL) == CL_SUCCESS) {
            printf("  Max Clock: %dMhz\n", uint_ret);
        }
        if (clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &bool_ret, NULL) == CL_SUCCESS) {
            printf("  Available: %d\n", bool_ret);
        }
        if (clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, sizeof(cl_bool), &bool_ret, NULL) == CL_SUCCESS) {
            printf("  Compiler Avail: %d\n", bool_ret);
        }
        if (clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &bool_ret, NULL) == CL_SUCCESS) {
            printf("  Img Support: %d\n", bool_ret);
            if (bool_ret) {

                if (clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &actual_size, NULL) == CL_SUCCESS) {
                    printf("   Max 2D height: %lu\n", actual_size);
                }
                if (clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &actual_size, NULL) == CL_SUCCESS) {
                    printf("   Max 2D width: %lu\n", actual_size);
                } 
            }

        }
        if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &actual_size, NULL) == CL_SUCCESS) {
            printf("  Max workgroup size: %lu\n", actual_size);
        }

        cl_command_queue_properties props;
        if (clGetDeviceInfo(dev, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &props, NULL) == CL_SUCCESS) {
            printf("  Supported queue properties: \n");
            if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                printf("    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
            if ((props & CL_QUEUE_PROFILING_ENABLE) == CL_QUEUE_PROFILING_ENABLE)
                printf("    CL_QUEUE_PROFILING_ENABLE\n");
        }


        printf("\n");
    }


    free(plats);


    return Py_BuildValue("i",0);
}

