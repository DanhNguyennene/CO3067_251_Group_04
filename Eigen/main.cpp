#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>


// Temporary function to multiply matrix
std::vector<std::vector<float>> multiplyMatrices(
    const std::vector<std::vector<float>>& mat1,
    const std::vector<std::vector<float>>& mat2) {

    int rows1 = mat1.size();
    int cols1 = mat1[0].size();
    int rows2 = mat2.size();
    int cols2 = mat2[0].size();

    if (cols1 != rows2) {
        throw std::runtime_error("Matrix dimensions are not compatible for multiplication.");
    }

    std::vector<std::vector<float>> result(rows1, std::vector<float>(cols2, 0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<float>> matrixToVector(Eigen::MatrixXd matrix){
    int size = matrix.rows();
    std::vector<std::vector<float>> result(size, std::vector<float>(size, 0));
    for(int i =0; i<size; i++){
        for(int j=0; j<size; j++){
            result[i][j]= matrix(i,j);
        }
    }
    return result;
}

Eigen::MatrixXd vectorToMatrix(std::vector<std::vector<float>> matrix){
    int size = matrix.size();
    Eigen::MatrixXd result = Eigen::MatrixXd(size, size);
    for(int i =0; i<size; i++){
        for(int j=0; j<size; j++){
            result(i,j)= matrix[i][j];
        }
    }
    return result;
}

int oneTest(int size, std::chrono::duration<double> time[2]){
    Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(size,size)*1000;
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(size,size)*1000;

    auto startTime = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd M3 = M1*M2;
    auto endTime= std::chrono::high_resolution_clock::now();
    time[0] = endTime- startTime;

    std::vector<std::vector<float>> M4 = matrixToVector(M1);
    std::vector<std::vector<float>> M5 = matrixToVector(M2);

    startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> M6 = multiplyMatrices(M4, M5);
    endTime= std::chrono::high_resolution_clock::now();
    time[1] = endTime- startTime;
    Eigen::MatrixXd M7 = vectorToMatrix(M6);
    // std::cout << M3<<"\n\n";
    // std::cout << M7<<"\n\n";
    return (M3.isApprox(M7));
    
}

int main(int argc, char *   argv[]){
    std::chrono::duration<double> time[2];
    int size = 1000;
    int iteration = 5;
    if(argc>2){
        size = std::atoi(argv[1]);
        iteration = std::atoi(argv[2]);

    }
    double timeEigen = 0.0;
    double timeOurFunction = 0.0;
    int nbError = 0;
    for(int i =0; i<iteration; i++){
        nbError+= oneTest(size, time);
        timeEigen+= time[0].count();
        timeOurFunction+= time[1].count();
    }
    double averageEigen = timeEigen/iteration; 
    double averageOurFunction = timeOurFunction/iteration; 
    std::cout << "Average time for our function " << averageOurFunction << " seconds with " << nbError << " error in " << iteration << " iterations\n";
    std::cout << "Average time for Eigan lib " << averageEigen << " seconds, " << averageOurFunction/averageEigen << " time faster\n";
    return 0;
}