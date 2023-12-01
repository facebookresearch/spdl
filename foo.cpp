#include <stdio.h>

int main(){
  printf("__has_include(<demangle.h>) -> %d\n", 
#if __has_include(<demangle.h>)
         1
#else
         0
#endif
 );
  printf("__has_include(<cxxabi.h>) -> %d\n", 
#if __has_include(<cxxabi.h>)
         1
#else
         0
#endif
 );
 return 0;
}
