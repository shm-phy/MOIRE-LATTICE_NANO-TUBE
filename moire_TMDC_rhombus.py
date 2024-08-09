
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

#import PyQt5
import os.path
import sys
from termcolor import colored
import argparse
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
#plt.switch_backend('Qt5Agg')
#print (plt.get_backend())

#####################################################################

def create_layer1(nx, ny, nx_ex, ny_ey, d_MZ, Mz, Zz_up, Zz_down, lx, ly):
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
    MZ2 = np.zeros([N, 4], float)

    MZ2[(ny*6 - 6) : (ny*6),1:] = unit
    MZ2[(ny*6 - 6) : (ny*6), 0] = [2, 1, 3, 2, 1, 3]
    y_incr = np.zeros([6, 4], float)
    y_incr[:, 2] = ly

    x_incr = np.zeros([ny*6, 4], float)
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


def cs_theta(m, r, lx):
    theta = np.arccos((3 * m*m + 3*m*r + 0.5 * r*r)/ (3 * m*m + 3*m*r + r*r)) * 180 / np.pi
    print()
    print("\tAngle of rotation (degree): ", theta)

    a1_x = lx
    a1_y = 0
    a2_x = lx / 2
    a2_y = lx * np.cos(np.pi/6)
    a1 = np.array([a1_x, a1_y], float)
    a2 = np.array([a2_x, a2_y], float)

    if np.gcd(r, 3) == 1:
        t1_x = m*a1_x + (m + r)*a2_x
        t1_y = m*a1_y + (m + r)*a2_y
        t2_x = -(m+r)*a1_x + (2*m + r)*a2_x
        t2_y = -(m+r)*a1_y + (2*m + r)*a2_y
        N = 6 * ((m+r)**2 + m*(2*m + r))
        print("\tNumber of atoms in moire cell from formula: ", N)
    elif np.gcd(r, 3) == 3:
        t1_x = (m+(r//3))*a1_x + (r//3)*a2_x
        t1_y = (m+(r//3))*a1_y + (r//3)*a2_y
        t2_x = -(r//3)*a1_x + (m + 2*(r//3))*a2_x
        t2_y = -(r//3)*a1_y + (m + 2*(r//3))*a2_y
        N = 2*((m+r)**2) + m * (r + 3*m)

    t1 = np.array([t1_x, t1_y], float)
    t2 = np.array([t2_x, t2_y], float)
    
    xv1 = 0
    yv1 = 0
    xv2 = t1_x
    yv2 = t1_y
    xv4 = t2_x
    yv4 = t2_y
    xv3 = xv2 + xv4
    yv3 = yv2 + yv4

    vertices = np.zeros([5, 2], float)
    vertices[:, 0] = [xv1, xv2, xv3, xv4, xv1]
    vertices[:, 1] = [yv1, yv2, yv3, yv4, yv1]
    
    Lx_sup = distance.euclidean(vertices[0, :], vertices[1, :])
    Ly_sup = distance.euclidean(vertices[0, :], vertices[3, :])
    print("\tLattice constant (x) of moire cell: ", Lx_sup/10, "nm")
    print("\tLattice constant (y) of moire cell: ", Ly_sup/10, "nm")
    print()
    
    return theta, N, Lx_sup, vertices


def create_layer2(layer1, m, r, H, v):

    def rotate(layer, cos, sin):
        layer_c = np.copy(layer)
        layer[:, 1] = layer_c[:, 1] * cos - layer_c[:, 2] * sin
        layer[:, 2] = layer_c[:, 1] * sin + layer_c[:, 2] * cos
        return layer

    layer2 = np.copy(layer1)
    layer2[:, 3] += H
    layer2[:, 0] += 3
    
    c = (3 * m*m + 3*m*r + 0.5 * r*r)/ (3 * m*m + 3*m*r + r*r)
    s = np.sqrt(1 - c*c)
    
    layer2 = rotate(layer2, c, s)

    ca = v[1, 0] - v[0, 0]
    co = v[1, 1] - v[0, 1]
    L = np.sqrt((co**2) + (ca**2))
    cos = ca / L
    sin = co / L

    x1v = v[0, 0]
    y1v = v[0, 1]
 
    layer1[:, 1] -= x1v
    layer1[:, 2] -= y1v
    
    layer1 = rotate(layer1, cos, -sin)
 
    layer2[:, 1] -= x1v
    layer2[:, 2] -= y1v
    
    layer2 = rotate(layer2, cos, -sin)
    
    v[:, 0] -= x1v
    v[:, 1] -= y1v
    v_c = np.copy(v)
    v[:, 0] = v_c[:, 0] * cos + v_c[:, 1] * sin
    v[:, 1] = -v_c[:, 0] * sin + v_c[:, 1] * cos
    unit = np.copy(v)
 
    return layer1, layer2, v, unit


def plot_rotated_sys(layer1, layer2, bond, d_MZ, N, count, vertices):
    print("\tPlotting rotated layers ...")
    M_m = layer2[0:N:3, 1:]
    Z_u = layer2[1:N:3, 1:]
    Z_dwn = layer2[2:N:3, 1:]
    
    plt.scatter(M_m[:, 0], M_m[:, 1], marker='.', color='b')
    plt.scatter(Z_u[:, 0], Z_u[:, 1], marker='.', color='y')
    
    M_m1 = layer1[0:N:3, 1:]
    Z_u1 = layer1[1:N:3, 1:]
    Z_dwn1 = layer1[2:N:3, 1:]
    
    plt.scatter(M_m1[:, 0], M_m1[:, 1], marker='.', color='b')
    plt.scatter(Z_u1[:, 0], Z_u1[:, 1], marker='.', color='y')
    
    plt.plot((layer2[bond[:count, 0], 1], layer2[bond[:count, 1], 1]), \
            (layer2[bond[:count, 0], 2], layer2[bond[:count, 1], 2]), '-', color='blue')

    plt.plot((layer1[bond[:count, 0], 1], layer1[bond[:count, 1], 1]), \
            (layer1[bond[:count, 0], 2], layer1[bond[:count, 1], 2]), '-', color='red')
 
    vertices_loc = np.copy(vertices)
    
    dx = d_MZ / 2
    dy = d_MZ / 2
    vertices_loc[:, 0] += dx
    vertices_loc[:, 1] -= dy
    plt.plot(vertices_loc[:, 0], vertices_loc[:, 1], '-', color = 'cyan')
    
    plt.plot(vertices[:, 0], vertices[:, 1], '-', color='black')
    plt.show()


def cut_layer(nx, ny, layer1, layer2, N, bond, count, vertices, unit):
    dl = d_MZ / 3
    layer1[:, 1] += dl
    layer1[:, 2] -= dl
    layer2[:, 1] += dl
    layer2[:, 2] -= dl
    unit[:, 0] += dl
    unit[:, 1] -= dl

    L_sup = distance.euclidean(vertices[0, :], vertices[1, :])
    Ly_sup = distance.euclidean(vertices[0, :], vertices[3, :])
    
    m1 = (vertices[1, 1] - vertices[0, 1]) / (vertices[1, 0] - vertices[0, 0])

    co = vertices[2, 1] - vertices[3, 1] 
    ca = vertices[2, 0] - vertices[3, 0]
    m2 = co / ca
    cos_phi = ca / np.sqrt(co**2 + ca**2)
    sin_phi = co / np.sqrt(co**2 + ca**2)

    co_v = vertices[2, 1] - vertices[1, 1]
    ca_v = vertices[2, 0] - vertices[1, 0] 
    cos_theta = ca_v / np.sqrt(co_v**2 + ca_v**2)
    sin_theta = co_v / np.sqrt(co_v**2 + ca_v**2)

    sin_t_p = sin_theta*cos_phi - cos_theta*sin_phi
    d = L_sup*sin_t_p
    ly_v = d / cos_phi
    def check_ver(xp, yp):
        yl = vertices[0, 1] + m1 * (xp - vertices[0, 0])
        yh = m1 * xp + (vertices[0, 1] - m1*vertices[0, 0] + ny*ly_v) 
        eps = 1e-6
        if (yp - yl) >= eps and (yh - yp) >= eps :
            return True
        else:
            return False
    
    lx_h = d / sin_theta
    ly_h = d / cos_theta
    mi1 = (vertices[3, 0] - vertices[0, 0]) / (vertices[3, 1] - vertices[0, 1])
    mi2 = (vertices[2, 0] - vertices[1, 0]) / (vertices[2, 1] - vertices[1, 1])
    def check_hr(xp, yp):
        xl = vertices[0, 0] + mi1 * (yp - vertices[0, 1])
        xh = mi2 * yp + nx*lx_h
        eps = 1e-6
        if (xp - xl) >= eps and (xh - xp) >= eps:
            return True
        else:
            return False
    
    def bond_cut(layer, bond, count):
        cut = np.zeros([N, 4], float)
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
    cut2, cnt2, bond2, b_c2 = bond_cut(layer2, bond, count) 

    v_vecxh = vertices[1, 0] - vertices[0, 0]
    v_vecyh = vertices[1, 1] - vertices[0, 1] 
    v_vecxv = vertices[3, 0] - vertices[0, 0]
    v_vecyv = vertices[3, 1] - vertices[0, 1]
    vertices[1:3, 0] += (nx-1)*v_vecxh
    vertices[1:3, 1] += (nx-1)*v_vecyh
    vertices[2:4, 0] += (ny-1)*v_vecxv
    vertices[2:4, 1] += (ny-1)*v_vecyv

    if (cnt1 != cnt2):
        print(colored("\t[ERROR]:", 'red'), "No. of atoms in layer1 and layer2 does not matches")
        print(colored("\t[SOLUTION]:", 'green'), "change the value of \"n_x\", \"n_y\", \"nx_ex\" and \"ny_ex\" \
inside the script (line no. 536 --> 539)")
        print()
    print("\tNumber of atoms in layer1 : ", cnt1)
    print("\tNumber of atoms in layer2 : ", cnt2)
    print("\tNumber of bonds in layer1 : ", b_c1)
    print("\tNumber of bonds in layer2 : ", b_c2)
    print()

    Lx_sup = distance.euclidean(vertices[0, :], vertices[1, :])
    Ly_sup = distance.euclidean(vertices[0, :], vertices[3, :])

    print("\tTotal number of atoms in twisted system: ", (cnt1+cnt2))
    print("\tTotal number of bonds in twisted system: ", (b_c1+b_c2))
    print("\tLength of the twisted system: ", Lx_sup/10, "nm")
    print("\tWidth of the twisted system: ", Ly_sup/10, "nm")
    print()
    
    return cut1, cnt1, cut2, cnt2, bond1, b_c1, bond2, b_c2, vertices, unit


def plot_twisted_system(layer1, cnt, layer2, cnt2, vertices, unit, bond1, b_c, bond2, b_c2, theta):
    print("\tPlotting twisted system ...")
    fig, ax = plt.subplots()
    S = 10.6 #8.6
    M1 = np.zeros([cnt, 2], float)
    Z1 = np.zeros([cnt, 2], float)
    M_count = 0
    Z_count = 0
    for i in range(cnt):
        if layer1[i, 0] == 2:
            M1[M_count, :] = layer1[i, 1:3]
            M_count += 1
        if layer1[i, 0] == 1 or layer1[i, 0] == 3:
            Z1[Z_count, :] = layer1[i, 1:3]
            Z_count += 1
    M1 = M1[:M_count, :]
    Z1 = Z1[:Z_count, :]
    
    M2 = np.zeros([cnt2, 2], float)
    Z2 = np.zeros([cnt2, 2], float)
    M_count = 0
    Z_count = 0
    for i in range(cnt2):
        if layer2[i, 0] == 5:
            M2[M_count, :] = layer2[i, 1:3]
            M_count += 1
        if layer2[i, 0] == 4 or layer2[i, 0] == 6:
            Z2[Z_count, :] = layer2[i, 1:3]
            Z_count += 1
    M2 = M2[:M_count, :]
    Z2 = Z2[:Z_count, :]
    
    ax.scatter(M1[:, 0], M1[:, 1], s=S, marker='o',color='r')
    ax.scatter(Z1[:, 0], Z1[:, 1], s=S, marker='o', color='b')
     
    ax.scatter(M2[:, 0], M2[:, 1], s=S, marker='o', color='r')
    ax.scatter(Z2[:, 0], Z2[:, 1], s=S, marker='o', color='b')
    
    ax.plot(vertices[:, 0], vertices[:, 1], '-', color='black')
    vecxh = unit[1, 0] - unit[0, 0]
    vecyh = unit[1, 1] - unit[0, 1]
    vecxv = unit[3, 0] - unit[0, 0]
    vecyv = unit[3, 1] - unit[0, 1]
    unit[:, 0] += (vecxh + vecxv)
    unit[:, 1] += (vecyh + vecyv)
    ax.plot(unit[:, 0], unit[:, 1], '-', color='blue')
    ax.fill(unit[:4, 0], unit[:4, 1], 'c', alpha=0.4)

    ax.plot((layer1[bond1[:b_c, 0], 1], layer1[bond1[:b_c, 1], 1]), \
            (layer1[bond1[:b_c, 0], 2], layer1[bond1[:b_c, 1], 2]), '-', linewidth=0.82, color='black')

    ax.plot((layer2[bond2[:b_c2, 0], 1], layer2[bond2[:b_c2, 1], 1]), \
            (layer2[bond2[:b_c2, 0], 2], layer2[bond2[:b_c2, 1], 2]), '-', linewidth=0.82, color='black')
    
    print()
    plt.show()
    ax.axis("off")
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    #flname = "twisted_"+str("{:.2f}".format(theta))+".svg"
    #plt.savefig(flname, format='svg', bbox_inches='tight', pad_inches=0)
    

def write_data(theta, L_sup, Nx, Ny, H, v, Zz_down, Zz_up, \
        layer1, cnt1, bond1, b_c1, layer2, cnt2, bond2, b_c2, bond_flag):

    directory = "./LAMMPS_DATA/Angle_"+str("{:.1f}".format(theta))+"/"
    filename = "data.MoS2_"+str(int(L_sup*Nx/10))+"_"+str(int(L_sup*Ny/10))
    file_path = os.path.join(directory, filename)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    layer = np.concatenate((layer1, layer2), axis=0)
    N_tot = cnt1 + cnt2

    f = open(file_path, "w+")
    f.write("LAMMPS Atom File\n\n")
    f.write("%d atoms\n" % N_tot)
    if bond_flag:
        f.write("%d bonds\n" % (b_c1+b_c2))
    else:
        f.write("0 bonds\n")
    f.write("0 angles\n")
    f.write("0 dihedrals\n")
    f.write("0 impropers\n\n")
    f.write("6 atom types\n")
    if bond_flag:
        f.write("1 bond types\n")
    else:
        f.write("0 bond types\n")
    f.write("0 angle types\n\n")
    f.write("%f %f xlo xhi\n" % (v[0, 0], v[1, 0]))
    f.write("%f %f ylo yhi\n" % (v[0, 1], v[3, 1]))
    f.write("%f %f zlo zhi\n" % ((Zz_down - 2*6.15), (Zz_up + H + 2*6.15)))
    f.write("%f %f %f xy xz yz\n\n" % (v[3, 0], 0.00, 0.00))
    f.write("Masses\n\n")
    f.write("   1 32.065\n")
    f.write("   2 95.940\n")
    f.write("   3 32.065\n")
    f.write("   4 32.065\n")
    f.write("   5 95.940\n")
    f.write("   6 32.065\n\n")
    f.write("Atoms\n\n")
    for k in range(N_tot):
        mol_tag = 1
        if k >= cnt1:
            mol_tag = 2
        f.write("   %d  %d    %d  0   %f  %f  %f\n"
                % (k+1, mol_tag, layer[k, 0], layer[k, 1],
                    layer[k, 2], layer[k, 3]))
    if bond_flag:
        f.write("\n")
        f.write("Bonds\n\n")
        for b in range(b_c1):
            f.write("   %d  1   %d  %d\n" % (b+1, bond1[b, 0]+1, bond1[b, 1]+1))
        bond2 += cnt1
        for b2 in range(b_c2):
            f.write("   %d  1   %d  %d\n" % (b2+1+b_c1, bond2[b2, 0]+1, bond2[b2, 1]+1))
    
    f.close()
    print("\tLAMMPS data file '",filename,"' created in", directory)
    print()


if __name__ == "__main__" :

    d_MZ = 1.817858
    d_MZ_z = 1.568711
    
    Zz_down = 2*6.15 
    Mz = Zz_down + d_MZ_z
    Zz_up = Mz + d_MZ_z
    
    lx = d_MZ * np.cos(np.pi/6) * 2
    ly = 3 * d_MZ
    
    nx_ex = 84 
    ny_ex = 84
    n_x = 124
    n_y = 122
    
    nx = nx_ex + n_x
    ny = ny_ex +  n_y
    N_tot = ny*6*nx
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-Nx", type=int, default=3, help="Give integer value of Nx")
    parser.add_argument("-Ny", type=int, default=3, help="Give integer value of Ny")
    parser.add_argument("-H", type=float, default=6.35, help="Enter ILS in Anstrom")
    parser.add_argument("-m", type=int, default=3, \
            help="Enter integer value of m for the commensurate angle")
    parser.add_argument("-r", type=int, default=1, \
            help="Enter integer value of r for the commensurate angle")
    parser.add_argument("--bond", dest='bond_flag', action='store_true', \
            help="Enter boolean value of bond_flag")
    parser.add_argument("--no-bond", dest='bond_flag', action='store_false', \
            help="Enter boolean value of bond_flag")
    parser.add_argument("--write", dest='write_flag', action='store_true', \
            help="Enter boolean value to write data in LAMMPS input format")
    parser.add_argument("--no-write", dest='write_flag', action='store_false', \
            help="Enter boolean value to write data in LAMMPS input format")
    parser.set_defaults(bond_flag=True, write_flag=True)
    args = parser.parse_args()
    Nx = args.Nx
    Ny = args.Ny
    H = args.H
    m = args.m
    r = args.r
    bond_flag = args.bond_flag
    write_flag = args.write_flag
    
    
    layer1 = create_layer1(nx, ny, nx_ex, ny_ex, d_MZ, Mz, Zz_up, Zz_down, lx, ly)
    
    bond, count = create_bond(nx, ny)
    
    theta, Num, L_sup, vertices = cs_theta(m, r, lx)
    
    layer1, layer2, vertices, unit = create_layer2(layer1, m, r, H, vertices)
    
    #plot_rotated_sys(layer1, layer2, bond, d_MZ, N_tot, count, vertices)
    
    layer1, cnt1, layer2, cnt2, bond1, b_c1 , bond2, b_c2, vertices, unit = \
            cut_layer(Nx, Ny, layer1, layer2, N_tot, bond, count, vertices, unit)
    
    plot_twisted_system(layer1, cnt1, layer2, cnt2, vertices, unit, bond1, b_c1, bond2, b_c2, theta)
    
    if cnt1 == cnt2 and write_flag:
        write_data(theta, L_sup, Nx, Ny, H, vertices, Zz_down, Zz_up, \
             layer1, cnt1, bond1, b_c1, layer2, cnt2, bond2, b_c2, bond_flag)


