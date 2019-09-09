#pragma ones
/*
Copyright (c) 2019, Sanaxen
All rights reserved.

Use of this source code is governed by a MIT license that can be found
in the LICENSE file.
*/
#define BOOL_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = (atoi(argv[i+1]) == 0) ? false : true;i++;continue;}
#define FLOAT_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = atof(argv[i+1]);i++;continue;}
#define INT_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = atoi(argv[i+1]);i++;continue;}
#define CSTR_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = argv[i+1];i++;continue;}
#define STR_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = std::string(argv[i+1]);i++;continue;}
#define HELP_OPT(i, var, opt) if (strcmp(argv[i], opt) == 0){var = true;continue;}
