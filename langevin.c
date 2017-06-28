#include <stdio.h>


int data0;
int data1;
// Data handling function


int main()

{
    const data_handle(const filename)
        
        size_t buffsize = 128;
        char *buffer;
        FILE *fp;
        fp = fopen(filename,'w+');
        // The file should be in csv format. A data handler which can detect the type and either convert it to csv or handle different formats should be implemented in the future.
        buffer = (char *)malloc(buffsize * sizeof(char))
        data0 = getline(buffer , buffsize, fp)

}
//double trajectory[steps][data0][data1];

//double potential (double x[data0][data1], double x0[data0][data1],
