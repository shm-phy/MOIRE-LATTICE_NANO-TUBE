#!/usr/bin/python3

# ============================================================================= #
#    Copyright (C) 2021  Soham Mandal                                           #
#                                                                               #
#    This program is free software: you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by       #
#    the Free Software Foundation, either version 3 of the License, or          #
#    (at your option) any later version.                                        #
#                                                                               #
#    This program is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#    GNU General Public License for more details.                               #
#                                                                               #
#    You should have received a copy of the GNU General Public License          #
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
#                                                                               #
#    e-mail: phy.soham@gmail.com                                                #
# ============================================================================= #

import os.path
import sys
from termcolor import colored
import argparse
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

#####################################################################

def create_layer1(nx, ny, nx_ex, ny_ey, d_MZ, Mz, Zz_up, Zz_down, lx, ly, shift, n, m):

    x1 = -lx * nx_ex
    y1 = -ly * ny_ex
    z1 = Zz_up
    x2 = x1 # -lx * nx_ex
    y2 = y1 # -ly * ny_ex
    z2 = Zz_down
    
    x3 = x1
    y3 = y1 + d_MZ
    z3 = Mz
    
    x4 = x1 + d_MZ * np.cos(np.pi/6)
    y4 = y3 + d_MZ / 2
    z4 = Zz_up
    x5 = x4
    y5 = y4
    z5 = Zz_down
    
    x6 = x4
    y6 = y4 + d_MZ
    z6 = Mz
    
    unit = np.array([[x6, y6, z6], [x5, y5, z5], 
            [x4, y4, z4], [x3, y3, z3], [x2, y2, z2], [x1, y1, z1]], float)

    N = ny*6*nx
    MZ2 = np.zeros([N, 5], float)

    MZ2[(ny*6 - 6) : (ny*6),1:4] = unit
    
    MZ2[(ny*6 - 6) : (ny*6), 0] = [1, 2, 3, 1, 2, 3]

    c = lx * np.sqrt(n**2 + n*m + m**2)
    D = c/np.pi
    R = D/2
    R_in = R - d_MZ_z
    R_out = R + d_MZ_z
    MZ2[(ny*6 - 6) : (ny*6), 4] = [R, R_in, R_out, R, R_in, R_out]

    y_incr = np.zeros([6, 5], float)
    y_incr[:, 2] = ly

    x_incr = np.zeros([ny*6, 5], float)
    x_incr[:, 1] = lx

    start_o = ny*6 - 6
    end_o = ny*6

    for i in range(ny - 1):
        start_n = start_o - 6
        end_n = start_o
        MZ2[start_n:end_n, :] = MZ2[start_o:end_o, :] + y_incr
        start_o = start_n
        end_o = end_n

    start_o = 0
    end_o = ny * 6
    
    for j in range(nx - 1):
        start_n = end_o
        end_n = start_n + ny*6
        MZ2[start_n:end_n, :] = MZ2[start_o : end_o, :] + x_incr
        start_o = start_n
        end_o = end_n

    if shift :
        shft = d_MZ / 2
        MZ2[:, 1] += shft
        MZ2[:, 2] += shft
    
    return MZ2


def create_bond(nx, ny):
    atm = 0
    count = 0
    bond = np.zeros([ny*6*nx*6, 2], int)
    
    for j in range(nx):
        for i in range(ny-1):
            bond[count, :] = [atm, atm+1]
            count += 1
            bond[count, :] = [atm, atm+2]
            count += 1
            bond[count, :] = [atm+1, atm+3]
            count += 1
            bond[count, :] = [atm+2, atm+3]
            count += 1
            atm += 3
            bond[count, :] = [atm, atm+1]
            count += 1
            bond[count, :] = [atm, atm+2]
            count += 1
            bond[count, :] = [atm+1, atm+3]
            count += 1
            bond[count, :] = [atm+2, atm+3]
            count += 1
            atm += 3
        bond[count, :] = [atm, atm+1]
        count += 1
        bond[count, :] = [atm, atm+2]
        count += 1
        bond[count, :] = [atm+1, atm+3]
        count += 1
        bond[count, :] = [atm+2, atm+3]
        count += 1
        atm += 3
        bond[count, :] = [atm, atm+1]
        count += 1
        bond[count, :] = [atm, atm+2]
        count += 1
        atm += 3
    Z1 = np.arange(1, ny*6, 6, int)
    (Z_n, ) = Z1.shape
    Z2 = np.arange(2, ny*6, 6, int) # or Z1+1
    Z_i = ny*6 + 2
    M = np.arange(6, ny*6, 6, int)
    (M_n, ) = M.shape
    M_i = ny*6 - 2
    
    for j in range(nx-1):
        Z_M = Z1 + Z_i
        bond[count:count+Z_n, 0] = Z1
        bond[count:count+Z_n, 1] = Z_M
        count += Z_n
        bond[count:count+Z_n, 0] = Z2
        bond[count:count+Z_n, 1] = Z_M
        count += Z_n
        M_Z1 = M + M_i
        M_Z2 = M_Z1 + 1
        bond[count:count+M_n, 0] = M
        bond[count:count+M_n, 1] = M_Z1
        count += M_n
        bond[count:count+M_n, 0] = M
        bond[count:count+M_n, 1] = M_Z2
        count += M_n
        Z1 += ny*6
        Z2 += ny*6
        M += ny*6  

    return bond, count


def specifications(n, m, lx, NL):

    c = lx * np.sqrt(n**2 + n*m + m**2)
    d = c/np.pi
    alpha = np.arctan2(m*np.sqrt(3)*0.5, (n+0.5*m)) 

    print()
    print("\talpha (degree): ", alpha*180/np.pi)

    a1_x = lx
    a1_y = 0
    a2_x = lx / 2
    a2_y = lx * np.cos(np.pi/6)
    u = np.array([a1_x, a1_y], float)
    v = np.array([a2_x, a2_y], float)

    w1 = n*u
    w2 = m*v
    w = w1 + w2

    w_prime = -(n+m)*u + n*v

    GCD = np.gcd((2*(n+m)-n), (2*n+m))
    nu = (n+2*m) // GCD
    nv = (2*n+m) // GCD
    w_prnd = -nu*u + nv*v

    print("\tMinimum length of Nanotube for PBC to satisfy: ",\
            distance.euclidean([0.0, 0.0], w_prnd), "Ang")
    
    L1 = NL * w_prnd 
    L2 = w + L1

    eps = 1E-8
    if (c - distance.euclidean([0, 0], w)) <= eps :
        print("\tCircumference of nanotube: ", c, "Ang")
        print("\tRadius of the nanotube: ", d/2, "Ang")

    vertices = np.zeros([5, 2], float)
    vertices[0, :] = [0.0, 0.0]
    vertices[1, :] = w
    vertices[2, :] = L2
    vertices[3, :] = L1
    vertices[4, :] = vertices[0, :]


    return alpha, vertices


def rotate_system(layer1, n, m, v):

    def rotate(layer, cos, sin):
        layer_c = np.copy(layer)
        layer[:, 1] = layer_c[:, 1] * cos - layer_c[:, 2] * sin
        layer[:, 2] = layer_c[:, 1] * sin + layer_c[:, 2] * cos
        return layer


    alpha = np.arctan2(m*np.sqrt(3)*0.5, (n+0.5*m)) 

    theta = -alpha
    c = np.cos(theta)
    s = np.sin(theta)

    layer1 = rotate(layer1, c, s)

    v_c = np.copy(v)
    v[:, 0] = v_c[:, 0] * c - v_c[:, 1] * s
    v[:, 1] = v_c[:, 0] * s + v_c[:, 1] * c

    return layer1, v


def plot_rotated_sys(layer1, bond, vertices, plot_bond, N):
    print("\tPlotting rotated layers ...")
    
    M_m1 = layer1[0:N:3, 1:]
    Z_u1 = layer1[1:N:3, 1:]
    Z_dwn1 = layer1[2:N:3, 1:]
    
    plt.scatter(M_m1[:, 0], M_m1[:, 1], marker='o', color='b')
    plt.scatter(Z_u1[:, 0], Z_u1[:, 1], marker='o', color='y')
    plt.scatter(Z_dwn1[:, 0], Z_dwn1[:, 1], marker='o', color='r')
    
    if plot_bond :
        plt.plot((layer1[bond[:count, 0], 1], layer1[bond[:count, 1], 1]), \
                (layer1[bond[:count, 0], 2], layer1[bond[:count, 1], 2]), '-', color='red')
 
    
    plt.plot(vertices[:, 0], vertices[:, 1], '-', color='black')
    plt.show()


def cut_layer(d_MZ, layer1, N, bond, count, vertices):
    dl = d_MZ / 3
    layer1[:, 1] += dl
    layer1[:, 2] -= dl

    Lx_sup = distance.euclidean(vertices[0, :], vertices[1, :])
    Ly_sup = distance.euclidean(vertices[0, :], vertices[3, :])

    def check_ver(xp, yp):
        yl = vertices[0, 1] 
        yh = vertices[3, 1]
        eps = 1e-6
        if (yp - yl) >= eps and (yh - yp) >= eps :
            return True
        else:
            return False
    
    def check_hr(xp, yp):
        xl = vertices[0, 0]
        xh = vertices[1, 0]
        eps = 1e-6
        if (xp - xl) >= eps and (xh - xp) >= eps:
            return True
        else:
            return False
    
    def bond_cut(layer, bond, count):
        cut = np.zeros([N, 5], float)
        cnt = 0
        atom_list = np.zeros(N, int)
        bond_l = np.zeros([count, 2], int)
        b_c = 0
        for b in range(count):
            atom1 = False
            atom2 = False
            x1 = layer[bond[b, 0], 1]
            x2 = layer[bond[b, 1], 1]
            y1 = layer[bond[b, 0], 2]
            y2 = layer[bond[b, 1], 2]
            if check_ver(x1, y1) and check_hr(x1, y1):
                if (bond[b, 0] in atom_list):
                    index_1, = np.where(atom_list == bond[b, 0])
                    index1 = index_1[0]
                else:
                    atom_list[cnt] = bond[b, 0]
                    index1 = cnt
                    cut[cnt, :] = layer[bond[b, 0], :]
                    cnt += 1
                atom1 = True
            if check_ver(x2, y2) and check_hr(x2, y2):
                if (bond[b, 1] in atom_list):
                    index_2, = np.where(atom_list == bond[b, 1])
                    index2 = index_2[0]
                else:
                    atom_list[cnt] = bond[b, 1]
                    index2 = cnt
                    cut[cnt, :] = layer[bond[b, 1], :]
                    cnt += 1
                atom2 = True
            if atom1 and atom2:
                bond_l[b_c, :] = [index1, index2]
                b_c += 1
  
        bond_l = bond_l[:b_c, :]
        cut = cut[0:cnt, :]
        return cut, cnt, bond_l, b_c
    
    cut1, cnt1, bond1, b_c1 = bond_cut(layer1, bond, count) 

    print()
    print("\tLength of the nano-tube: ", Ly_sup, "Ang")
    print("\tNumber of atoms : ", cnt1)
    print("\tNumber of bonds : ", b_c1)
    print()

    return cut1, cnt1, bond1, b_c1

    
def roll_up_plane(layer1, vertices, N, adjust_center, vaccum, d_MZ_z) :
    
    c = distance.euclidean(vertices[1, :], vertices[0, :])
    D = c/np.pi
    R = D/2

    layer1[:, 1] -= c/2
    layer1[:, 3] += (R - d_MZ_z)

    layer_x = layer1[:, 1]
    Radius = layer1[:, 4]
    theta = layer_x / R

    layer1[:, 1] = Radius * np.sin(theta)
    layer1[:, 3] = Radius * np.cos(theta)

    v = np.zeros([4, 3], float)
    #origin
    v[0, 0] = -R
    v[0, 2] = -R
    v[0, 1] = vertices[0, 1]

    #x-direction
    v[1, 0] = R
    v[1, 2] = -R
    v[1, 1] = vertices[0, 1]
    
    #y-direction
    v[2, 0] = -R
    v[2, 2] = -R
    v[2, 1] = vertices[3, 1]

    #z-direction
    v[3, 0] = -R
    v[3, 2] = R
    v[3, 1] = vertices[0, 1]
    
    if adjust_center :

        layer1[:, 1] += R + vaccum
        layer1[:, 3] += R + vaccum

        v[:, 0] += R
        v[:, 2] += R
        #v[1, 0] += 2*vaccum
        #v[3, 2] += 2*vaccum
        vacumx = np.min(layer1[:, 1])
        vacumy = np.min(layer1[:, 3])
        v[1, 0] = np.max(layer1[:, 1]) + vacumx
        v[3, 2] = np.max(layer1[:, 3]) + vacumy

    return layer1, v


def write_data(n, m, layer, v, N_tot, bond1, b_c1, label, atomic_no, lammps_flag, siesta_flag):

    ly = (v[2, 1] - v[0, 1])/10

    if lammps_flag :
        directory = "./LAMMPS_DATA/tmdc-NT_"+str(n)+"_"+str(m)+"/"
        filename = "data.tmdc-NT-"+str(n)+"_"+str(m)+"-"+str("{:.1f}".format(ly))
        file_path = os.path.join(directory, filename)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        f = open(file_path, "w+")
        f.write("LAMMPS Atom File\n\n")
        f.write("%d atoms\n" % N_tot)
        if bond_flag:
            f.write("%d bonds\n" % (b_c1))
        else:
            f.write("0 bonds\n")
        f.write("0 angles\n")
        f.write("0 dihedrals\n")
        f.write("0 impropers\n\n")
        f.write("3 atom types\n")
        if bond_flag:
            f.write("1 bond types\n")
        else:
            f.write("0 bond types\n")
        f.write("0 angle types\n\n")

        f.write("%f %f xlo xhi\n" % (v[0, 0], v[1, 0]))
        f.write("%f %f ylo yhi\n" % (v[0, 1], v[2, 1]))
        f.write("%f %f zlo zhi\n\n" % (v[0, 2], v[3, 2]))

        f.write("Masses\n\n")
        f.write("   1 95.940\n")
        f.write("   2 32.065\n")
        f.write("   3 32.065\n\n")
        f.write("Atoms\n\n")
        mol_tag = 1
        N_M = 0
        N_Zdown = 0
        N_Zup = 0
        atm_ID = 0
        for k in range(N_tot):
            if layer[k, 0] == 1:
                N_M += 1
                atm_ID = 1
                mol_tag = 1
            if layer[k, 0] == 2:
                N_Zdown += 1
                atm_ID = 2
                mol_tag = 2
            if layer[k, 0] == 3:
                N_Zup += 1
                atm_ID = 2
                mol_tag = 3

            f.write("   %d  %d    %d  0   %f  %f  %f\n"
                    % (k+1, mol_tag, atm_ID, layer[k, 1],
                        layer[k, 2], layer[k, 3]))
        if bond_flag:
            f.write("\n")
            f.write("Bonds\n\n")
            for b in range(b_c1):
                f.write("   %d  1   %d  %d\n" % (b+1, bond1[b, 0]+1, bond1[b, 1]+1))
        
        f.close()
        if not((N_Zdown+N_Zup)/N_M == 2) :
            print(colored("\t[ERROR]:", 'red'), "Number of Mo and S are not commensurate")
        print("\tLAMMPS data file '",filename,"' created in", directory)
        print()

    if siesta_flag :
        directory = "./SIESTA_DATA/tmdc-NT_"+str(n)+"_"+str(m)+"/"
        filename = "tmdc-NT-"+str(n)+"_"+str(m)+"-"+str("{:.1f}".format(ly))+".fdf"
        file_path = os.path.join(directory, filename)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        f2 = open(file_path, "w+")

        f2.write("NumberOfSpecies           %d\n" % (len(label)))
        f2.write("NumberOfAtoms             %d\n" % N_tot)

        f2.write("%block ChemicalSpeciesLabel\n")
        for l in range(len(label)) :
            f2.write("  %d  %d  %s\n" %(l+1, atomic_no[l], label[l]))
        f2.write("%endblock ChemicalSpeciesLabel\n")
        f2.write("\n")

        if not(abs(np.sum(v[0, :])) <= 1.E-8) :
            print(colored("\t[WARNING]:", 'red'), "Origin of the box is not at 0")
            print("\tOrigin: ", v[0, :])

        box = np.array([(v[1, 0] - v[0, 0]), (v[2, 1] - v[0, 1]), \
                (v[3, 2] - v[0, 2])], float)
        ALAT = np.min(box)
        f2.write("LatticeConstant           %f Ang\n" %ALAT)
        f2.write("\n")
        f2.write("%block LatticeVectors\n")
        f2.write("   %f  %f  %f\n" % (box[0]/ALAT, 0.00, 0.00))
        f2.write("   %f  %f  %f\n" % (0.00, box[1]/ALAT, 0.00))
        f2.write("   %f  %f  %f\n" % (0.00, 0.00, box[2]/ALAT))
        f2.write("%endblock LatticeVectors\n")
        f2.write("\n")

        layer = layer[layer[:, 0].argsort()]

        f2.write("AtomicCoordinatesFormat Ang\n")
        f2.write("\n")
        f2.write("%block AtomicCoordinatesAndAtomicSpecies\n")
        line_format = "{x: <12s}   {y: <12s}   {z: <12s}   {typ: <4s} {atmNo: <4s} {typlbl: <4s}\n"
        accu_float = 3+1+5

        N_M = 0
        N_Z = 0
        for j in range(N_tot) :
            atm_ID = layer[j, 0]
            if atm_ID == 1 :
                lbl = "Mo"
                atm_typ = 1
                N_M += 1
            else :
                lbl = "S"
                atm_typ = 2
                N_Z += 1
            f2.write(line_format.format(x=str(layer[j, 1])[:accu_float], \
                    y=str(layer[j, 2])[:accu_float], z=str(layer[j, 3])[:accu_float], \
                    typ=str(int(atm_typ)), atmNo=str(j+1), typlbl=lbl))

        f2.write("%endblock AtomicCoordinatesAndAtomicSpecies\n")
        f2.write("\n")

        f2.close()
        if not(N_Z/N_M == 2) :
            print(colored("\t[ERROR]:", 'red'), "Number of Mo and S are not commensurate")

        print("\tSIESTA input data file crated at '", file_path, "'")
        print()



if __name__ == "__main__" :

    d_MZ = 1.817858
    d_MZ_z = 1.568711
    
    Zz_down = 0.0 #2*6.15 
    Mz = Zz_down + d_MZ_z
    Zz_up = Mz + d_MZ_z
    
    lx = d_MZ * np.cos(np.pi/6) * 2
    ly = 3 * d_MZ
    
    nx_ex = 434 
    ny_ex = 44
    n_x = 324
    n_y = 322
    
    #for testing
    #nx_ex = 34 
    #ny_ex = 4
    #n_x = 24
    #n_y = 22

    nx = nx_ex + n_x
    ny = ny_ex + n_y
    N_tot = ny*6*nx
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=8, \
            help="Enter integer value of m for the commensurate angle")
    parser.add_argument("-m", type=int, default=0, \
            help="Enter integer value of r for the commensurate angle")
    parser.add_argument("-NL", type=int, default=4, help="Give integer value of NL (Number of unit cell along length)")
    parser.add_argument("-vac", type=float, default=50.0, \
            help="Enter the amount of vaccum from NT wall required")
    
    parser.add_argument("--shift", dest='shift_flag', action='store_true', \
            help="Enter boolean value of shift_flag")
    parser.add_argument("--no-shift", dest='shift_flag', action='store_false', \
            help="Enter boolean value of shift_flag")
    
    parser.add_argument("--bond", dest='bond_flag', action='store_true', \
            help="Enter boolean value of bond_flag")
    parser.add_argument("--no-bond", dest='bond_flag', action='store_false', \
            help="Enter boolean value of bond_flag")

    parser.add_argument("--write-lammps", dest='lammps_flag', action='store_true', \
            help="Enter boolean value to write data in LAMMPS input format")
    parser.add_argument("--no-write", dest='lammps_flag', action='store_false', \
            help="Enter boolean value to write data in LAMMPS input format")
    parser.add_argument("--write-siesta", dest='siesta_flag', action='store_true', \
            help="Enter boolean value to write data in SIESTA input format")


    parser.set_defaults(bond_flag=True, shift_flag=True, lammps_flag=True, siesta_flag=True)

    args = parser.parse_args()
    NL = args.NL
    n = args.n
    m = args.m
    vaccum = args.vac
    shift_flag = args.shift_flag
    bond_flag = args.bond_flag
    lammps_flag = args.lammps_flag
    siesta_flag = args.siesta_flag

    adjust_center = True
    label = ["Mo", "S"]
    atomic_no = [42, 16]
    #plot_bond = True

    layer1 = create_layer1(nx, ny, nx_ex, ny_ex, d_MZ, Mz, Zz_up, Zz_down, lx, ly, shift_flag, n, m)

    bond, count = create_bond(nx, ny)
    
    alpha, vertices = specifications(n, m, lx, NL)
    
    layer1, vertices = rotate_system(layer1, n, m, vertices)
    
    layer1, N_tot, bond, N_bond = cut_layer(d_MZ, layer1, N_tot, bond, count, vertices)
    
    layer1, vertices = roll_up_plane(layer1, vertices, N_tot, adjust_center, vaccum, d_MZ_z)
   
    write_data(n, m, layer1, vertices, N_tot, bond, N_bond, label, atomic_no, lammps_flag, siesta_flag)

    #plot_rotated_sys(layer1, bond, vertices, plot_bond, N_tot)
  
