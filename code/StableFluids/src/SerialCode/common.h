namespace StableFluids {

	int find_option(int argc, char** argv, const char* option);
	int read_int(int argc, char** argv, const char* option, int default_value);
	float read_float(int argc, char** argv, const char* option, float default_value);
	char* read_string(int argc, char** argv, const char* option, char* default_value);
}
