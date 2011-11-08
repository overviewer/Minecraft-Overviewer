

#include "overviewer.h"


#include <OpenCL/OpenCL.h>

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


    return "UNKNOWN";
}



PyObject *
do_cl_init(PyObject *self, PyObject *args)
{
    /* Get the device that we will use */

    cl_device_id    work_device;
    cl_int error;
    if (clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &work_device, NULL) != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clGetDeviceIDs");
        return NULL;
    }

    context =clCreateContext(0, 1, &work_device, NULL, NULL, &error);
    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clCreateContext");
        return NULL;
    }

    queue = clCreateCommandQueue(context, work_device, 0, &error);
    if (error != CL_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed clCreateCommandQueue");
        return NULL;
    }    



    const char *program_name = "/Users/achin/devel/overviewer-readonly/paste.cl";
    struct stat s;
    if (stat(program_name,&s) != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed stat paste.cl");
        return NULL;

    }


    FILE* prog_file = fopen(program_name, "r");
    size_t filesize = s.st_size+1;
    char * source = (char*)malloc(filesize);
    bzero(source, filesize);
    if (fread(source, 1, filesize-1, prog_file) != filesize-1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Failed read all of paste.cl");
        return NULL;
    }
    fclose(prog_file);

    const size_t len = strlen(source);

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
        printf("error: %s\n", getClError(error));

        size_t logSize;
        clGetProgramBuildInfo(program, work_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *buildLog = malloc(logSize);
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




PyObject *
print_cl_info(PyObject *self, PyObject *args)
{

    cl_uint num_devs, num_plats;
    cl_int rc;

    clGetPlatformIDs(0, NULL, &num_plats);
    printf("Total number of platforms: %d\n", num_plats);


    cl_platform_id *plats = malloc(sizeof(cl_platform_id) *num_plats);
    clGetPlatformIDs(num_plats, plats, 0);
    int i;
    for (i = 0; i < num_plats; i++) {
        printf("Platform #%d\n", i);
        printf("-----------\n");
        cl_platform_id plat = plats[i];
        char name[120];

        if (clGetPlatformInfo(plat, CL_PLATFORM_NAME, 120, name, NULL) == CL_SUCCESS) {
            printf("  Name: %s\n", name);

        }
        if (clGetPlatformInfo(plat, CL_PLATFORM_VENDOR, 120, name, NULL) == CL_SUCCESS) {
            printf("  Vendor: %s\n", name);

        }
        if (clGetPlatformInfo(plat, CL_PLATFORM_VERSION, 120, name, NULL) == CL_SUCCESS) {
            printf("  Version: %s\n", name);
        }
    }
    printf("\n");

    rc = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs);
    printf("Total number of devices: %d\n", num_devs);

    cl_device_id *ids = malloc(sizeof(cl_device_id)*num_devs);
    rc = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devs, ids, NULL);

    for (i = 0; i < num_devs; i++ ){
        printf("Device #%d\n", i);
        printf("---------\n");
        cl_device_id dev = ids[i];
        char name[120];
        cl_device_type type;
        cl_uint uint_ret;
        cl_bool bool_ret;
        size_t actual_size;

        if (clGetDeviceInfo(dev, CL_DEVICE_NAME, 120, name, &actual_size) == CL_SUCCESS)
            printf("  Name: %s\n", name);

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
        printf("\n");
    }


    free(plats);


    return Py_BuildValue("i",0);
}

