#include "sapch.h"
#include <spdlog/spdlog.h>

#include <stdlib.h>
#include <string.h>
#include "common.h"
//
//  command line option processing
//
namespace StableFluids {

    int find_option(int argc, char** argv, const char* option) {
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], option) == 0)
                return i;
        return -1;
    }


    int read_int(int argc, char** argv, const char* option, int default_value) {
        int iplace = find_option(argc, argv, option);
        if (iplace >= 0 && iplace < argc - 1)
            return atoi(argv[iplace + 1]);
        return default_value;
    }

    float read_float(int argc, char** argv, const char* option, float default_value) {
        int iplace = find_option(argc, argv, option);
        if (iplace >= 0 && iplace < argc - 1)
            return atof(argv[iplace + 1]);
        return default_value;
    }

    char* read_string(int argc, char** argv, const char* option, char* default_value) {
        int iplace = find_option(argc, argv, option);
        if (iplace >= 0 && iplace < argc - 1)
            return argv[iplace + 1];
        return default_value;
    }
}
  