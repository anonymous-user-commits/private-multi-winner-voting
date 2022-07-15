#ifdef _MSC_VER
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL
#endif

#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL long long argmax(int party, int port,
                               std::vector<long long> &array);

#ifdef __cplusplus
}
#endif