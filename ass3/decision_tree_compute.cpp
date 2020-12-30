#include <iostream>
#include <cmath>
#include <vector>
using namespace std;
double ent(int x1, int x2){
    if (x1 == 0 && x2 == 0) return 0;
    double r1 = (double)x1/(x1+x2);
    double r2 = (double)x2/(x1+x2);
    if (x1 == 0) return -r2*log2(r2);
    else if (x2 == 0) return -r1*log2(r1);
    return -(r1*log2(r1) + r2*log2(r2));
}
int main(){
    //vector <int> v = {0,1,1,0};
    vector <int> v = {0,0,1,1};
    vector <int> numof1(v.size());
    vector <int> numof0(v.size());
    if (v[0] == 0){
        numof0[0] = 1;
        numof1[0] = 0;
    }
    else{
        numof0[0] = 0;
        numof1[0] = 1;
    }
    for (int i = 1; i < v.size(); ++i){
        if (v[i] == 0) {
            numof0[i] = numof0[i-1]+1;
            numof1[i] = numof1[i-1];
        }
        else{
            numof1[i] = numof1[i-1]+1;
            numof0[i] = numof0[i-1];
        }
    }
    int all = v.size();
    double m = 10000000;
    int index = 0;
    
    for (int i = 0; i < v.size()-1; ++i){
        double _ent = (double)(i+1)/all*ent(numof0[i], numof1[i]) + (double)(all-i-1)/all*ent(numof0[v.size()-1]-numof0[i], numof1[v.size()-1]-numof1[i]);
        cout << "index = " <<i<<"    " <<(double)(i+1)/all << "*ent("<<numof0[i]<<", "<<numof1[i]<<") + " 
        <<(double)(all-i-1)/all << "*ent("<<numof0[v.size()-1]-numof0[i]<<", "<<numof1[v.size()-1]-numof1[i] << ") = " << _ent<<endl;
        if (_ent < m){
            m = _ent;
            index = i;
        }
    }
    cout << index << ' ' << m << endl;
    return 0;
}