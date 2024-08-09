
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

def create_layer1(nx, ny, nx_ex, ny_ey, d_BN, hBN_z, lx, ly):

    x1 = -lx * nx_ex    #N
    y1 = -ly * ny_ex    #N
    z1 = hBN_z          #N
    
    x2 = x1             #B
    y2 = y1 + d_BN      #B
    z2 = hBN_z          #B
    
    x3 = x2 + d_BN * np.cos(np.pi/6)    #N
    y3 = y2 + d_BN / 2                  #N
    z3 = hBN_z                          #N
    
    x4 = x3                             #B
    y4 = y3 + d_BN                      #B
    z4 = hBN_z                          #B
    
    unit = np.array([[x4, y4, z4], [x3, y3, z3], [x2, y2, z2], [x1, y1, z1]], float)

    N = ny*4*nx
    Gr = np.zeros([N, 4], float)

    Gr[(ny*4 - 4) : (ny*4),1:] = unit
    Gr[(ny*4 - 4) : (ny*4), 0] = [1, 2, 1, 2]
    y_incr = np.zeros([4, 4], float)
    y_incr[:, 2] = ly

    x_incr = np.zeros([ny*4, 4], float)
    x_incr[:, 1] = lx

    start_o = ny*4 - 4
    end_o = ny*4

    for i in range(ny - 1):
        start_n = start_o - 4
        end_n = start_o
        Gr[start_n:end_n, :] = Gr[start_o:end_o, :] + y_incr
        start_o = start_n
        end_o = end_n

    start_o = 0
    end_o = ny * 4
    
    for j in range(nx - 1):
        start_n = end_o
        end_n = start_n + ny*4
        Gr[start_n:end_n, :] = Gr[start_o : end_o, :] + x_incr
        start_o = start_n
        end_o = end_n
    
    return Gr


def create_bond(nx, ny):
    atm = 0
    count = 0
    bond = np.zeros([ny*6*nx*6, 2], int)

    for j in range(nx):
        for i in range(ny-1):
            bond[count, :] = [atm, atm+1]
            count += 1
            bond[count, :] = [atm+1, atm+2]
            count += 1
            bond[count, :] = [atm+2, atm+3]
            count += 1
            bond[count, :] = [atm+3, atm+4]
            count += 1
            atm += 4
        bond[count, :] = [atm, atm+1]
        count += 1
        bond[count, :] = [atm+1, atm+2]
        count += 1
        bond[count, :] = [atm+2, atm+3]
        count += 1
        atm += 4
    C1 = np.arange(1, ny*4, 4, int)
    (C1_n, ) = C1.shape
    C1_i = ny*4 + 1
    C2 = np.arange(4, ny*4, 4, int)
    (C2_n, ) = C2.shape
    C2_i = ny*4 - 1

    for j in range(nx-1):
        C1_C = C1 + C1_i
        bond[count:count+C1_n, 0] = C1
        bond[count:count+C1_n, 1] = C1_C
        count += C1_n
        C2_C = C2 + C2_i
        bond[count:count+C2_n, 0] = C2
        bond[count:count+C2_n, 1] = C2_C
        count += C2_n
        C1 += ny*4
        C2 += ny*4
    bond = bond[:count, :]

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
        N = 4 * ((m+r)**2 + m*(2*m + r))
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
    print("\tLattice constant of hexagonal moire cell: ", distance.euclidean(vertices[0, :], vertices[1, :])/10, "nm")
    
    return theta, N, vertices



def create_layer2(layer1, m, r, H, v):

    def rotate(layer, cos, sin):
        layer_c = np.copy(layer)
        layer[:, 1] = layer_c[:, 1] * cos - layer_c[:, 2] * sin
        layer[:, 2] = layer_c[:, 1] * sin + layer_c[:, 2] * cos
        return layer

    layer2 = np.copy(layer1)
    layer2[:, 3] += H
    layer2[:, 0] += 2
    
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

    c1 = v[1, :]
    c2_p = v[3, :] - v[0, :]
    c2 = 2 * c2_p - c1
    v[1, :] = c1
    v[2, :] = v[1, :] + c2
    v[3, :] = c2
    unit = np.copy(v)

    Lx_sup = distance.euclidean(v[0, :], v[1, :])
    Ly_sup = distance.euclidean(v[0, :], v[3, :])
    print("\tLattice constant (x) of rectangular unit cell: ", Lx_sup/10, "nm")
    print("\tLattice constant (y) of rectangular unit cell: ", Ly_sup/10, "nm")
    print()
    
    unit = np.copy(v)
 
    return layer1, layer2, v, Lx_sup, Ly_sup, unit

def plot_rotated_sys(layer1, layer2, bond, d_BN, N, count, vertices):
    print("\tPlotting rotated layers ...")
    C2 = layer2[0:N, 1:]
    
    plt.scatter(C2[:, 0], C2[:, 1], marker='.', color='b')
    
    C1 = layer1[0:N:3, 1:]
    
    plt.scatter(C1[:, 0], C1[:, 1], marker='.', color='b')
    
    plt.plot((layer2[bond[:count, 0], 1], layer2[bond[:count, 1], 1]), \
            (layer2[bond[:count, 0], 2], layer2[bond[:count, 1], 2]), '-', color='blue')

    plt.plot((layer1[bond[:count, 0], 1], layer1[bond[:count, 1], 1]), \
            (layer1[bond[:count, 0], 2], layer1[bond[:count, 1], 2]), '-', color='red')
 
    vertices_loc = np.copy(vertices)
    
    dx = d_BN / 2
    dy = d_BN / 2
    vertices_loc[:, 0] += dx
    vertices_loc[:, 1] -= dy
    plt.plot(vertices_loc[:, 0], vertices_loc[:, 1], '-', color = 'cyan')
    
    plt.plot(vertices[:, 0], vertices[:, 1], '-', color='black')


def cut_layer(nx, ny, d_BN, layer1, layer2, N, bond, count, vertices, unit):
    dl = d_BN / 3
    layer1[:, 1] += dl
    layer1[:, 2] -= dl
    layer2[:, 1] += dl
    layer2[:, 2] -= dl
    unit[:, 0] += dl
    unit[:, 1] -= dl

    L_sup = distance.euclidean(vertices[0, :], vertices[1, :])
    Ly_sup = distance.euclidean(vertices[0, :], vertices[3, :])

    def check_ver(xp, yp):
        yl = vertices[0, 1] 
        yh = ny*Ly_sup
        eps = 1e-6
        if (yp - yl) >= eps and (yh - yp) >= eps :
            return True
        else:
            return False
    
    def check_hr(xp, yp):
        xl = vertices[0, 0]
        xh = nx*L_sup
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
inside the script (line no. 450 --> 453)")
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


def plot_twisted_system(layer1, cnt, layer2, cnt2, vertices, unit, bond1, b_c, bond2, b_c2):
    print("\tPlotting twisted system ...")
    fig, ax = plt.subplots()
    S = 8.6
    C_1 = layer1[:, 1:3]
    C_2 = layer2[:, 1:3]

    ax.scatter(C_1[:, 0], C_1[:, 1], s=S, marker='o',color='b')

    ax.scatter(C_2[:, 0], C_2[:, 1], s=S, marker='o', color='r')

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
    # frame1 = plt.gca()
    # frame1.axes.get_xaxis().set_visible(False)
    # frame1.axes.get_yaxis().set_visible(False)
    # plt.savefig('twisted_9.43.png', dpi=1200)
    

def write_data(theta, Lx_sup, Ly_sup, Nx, Ny, H, v, hBN_z, \
        layer1, cnt1, bond1, b_c1, layer2, cnt2, bond2, b_c2, bond_flag):

    directory = "./LAMMPS_DATA_BOND/Angle_"+str("{:.1f}".format(theta))+"/"
    filename = "data.hBN_rect_"+str(int(Lx_sup*Nx/10))+"_"+str(int(Ly_sup*Ny/10))
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
    f.write("4 atom types\n")
    if bond_flag:
        f.write("1 bond types\n")
    else:
        f.write("0 bond types\n")
    f.write("0 angle types\n\n")
    f.write("%f %f xlo xhi\n" % (v[0, 0], v[1, 0]))
    f.write("%f %f ylo yhi\n" % (v[0, 1], v[3, 1]))
    f.write("%f %f zlo zhi\n\n" % ((hBN_z - 4*1.7), (hBN_z + H + 4*1.7)))
    f.write("Masses\n\n")
    f.write("   1 10.811\n")
    f.write("   2 14.0067\n")
    f.write("   3 10.811\n")
    f.write("   4 14.0067\n\n")
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

    d_BN = 1.32 #Angstrom
    
    hBN_z = 4*1.7
    
    lx = d_BN * np.cos(np.pi/6) * 2
    ly = 3 * d_BN
    
    nx_ex = 84
    ny_ex = 84
    n_x = 84
    n_y = 82
    
    nx = nx_ex + n_x
    ny = ny_ex + n_y
    N_tot = ny*6*nx
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-Nx", type=int, default=4, help="Give integer value of Nx")
    parser.add_argument("-Ny", type=int, default=3, help="Give integer value of Ny")
    parser.add_argument("-H", type=float, default=6.0, help="Enter ILS in Anstrom")
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
    
    
    layer1 = create_layer1(nx, ny, nx_ex, ny_ex, d_BN, hBN_z, lx, ly)
    
    bond, count = create_bond(nx, ny)
    
    theta, Num, vertices = cs_theta(m, r, lx)
    
    layer1, layer2, vertices, Lx_sup, Ly_sup, unit = create_layer2(layer1, m, r, H, vertices)
    
    #plot_rotated_sys(layer1, layer2, bond, d_MoS, N_tot, count, vertices)
    
    layer1, cnt1, layer2, cnt2, bond1, b_c1 , bond2, b_c2, vertices, unit = \
            cut_layer(Nx, Ny, d_BN, layer1, layer2, N_tot, bond, count, vertices, unit)
    
    #plot_twisted_system(layer1, cnt1, layer2, cnt2, vertices, unit, bond1, b_c1, bond2, b_c2)
    
    if cnt1 == cnt2 and write_flag:
        write_data(theta, Lx_sup, Ly_sup, Nx, Ny, H, vertices, hBN_z, \
             layer1, cnt1, bond1, b_c1, layer2, cnt2, bond2, b_c2, bond_flag)
    
    
