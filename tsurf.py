import numpy as np
import sys
from scipy import interpolate

def n(m, a):
  G = 6.67e-8
  M_J = 1.898e30
  return np.sqrt(G*(M_J + m) / (a**3))

def k2_f(m, r):
  G = 6.67e-8
  g = (G*m)/pow(r, 2)
  rho_s = 3*m/(4*np.pi*pow(r, 3))
  mu = (19./2.) * (5e11)/(rho_s * g * r)
  return 1.5/(1 + mu)

def E_tidal(a, e, m, r):
  G = 6.67e-8
  Q = 100
  M_J = 1.898e30
  return (21. / 2) * G * k2_f(m, r) * (M_J**2) * (r**5) * n(m, a) * (e**2) / (Q * (a**6))

def T_eff4(a, e, m, r):
  sigma_B = 5.6704e-5
  epsilon_r = 0.9
  return E_tidal(a, e, m, r) / (4 * np.pi * sigma_B * epsilon_r * r**2)

def H(T, g):
  k_B = 1.3806e-16                  # cm^2 * g / (s^2 * K)  
  mu_CO2 = 44.01                    # g / cm^3
  m_H = 1.67e-24                    # g 
  return k_B * T / (mu_CO2 * m_H * g)

def get_kappa(p, T, f):
  if (p < 1e-5): p = 1e-5
  if (T < 50): T = 50
  if (T > 647.3): T = 647.3
  return 10**f(np.log10(p * 1e5), T)[0] * 10

def T_profile(a0, e0, m, p0, fit_kappa):
  D = 1.5
  G = 6.67e-8
  rho = 2.5791                        #average density of the Galilean system
  r = pow(3 * m / (4 * np.pi * rho), 1./3)
  convective_switch = 0
  g = G * m / (r**2)
  T_effective = (T_eff4(a0, e0, m, r))**(0.25)
  Delta = 0.2222                    # for CO2

  T_rad = pow(2,-0.25) * T_effective
  H_max = H(T_rad, g)
  z_max = - H_max * np.log(1e-5 / p0)
  delta_z = z_max / 1000

  p_rad = p0 * np.exp(-z_max / H(T_rad, g))        # Radiative pressure at the top
  T_prev = T_rad
  p_prev = p_rad
  optical_depth = 0
  z = z_max

  k_B = 1.3806e-16
  mu_CO2 = 44.01
  m_H = 1.67e-24
  density = (p_rad*1e6) / (k_B*T_rad) * (mu_CO2 * m_H)

  while(p_prev < p0):
    z = z - delta_z
    p_rad = p0 * np.exp(- z / H(T_rad, g))                                     # Radiative pressure
    T_rad = T_effective * pow((0.5 * (1 + D * optical_depth)), 0.25)           # Radiative temperature
    grad = (np.log(T_rad) - np.log(T_prev)) / (np.log(p_rad) - np.log(p_prev))

    if (grad < Delta):
      kr = get_kappa(p_rad, T_rad, fit_kappa)                      # cm^2 / g (pressure from bars to Pa)
      optical_depth = optical_depth + (kr * (p_rad - p_prev)*1e6 / g)[0]
      grad_opt = (kr * (p_rad - p_prev)*1e6 / g)[0] / delta_z
      T_prev = T_rad
      p_prev = p_rad

    else:

      if(convective_switch == 0):
        #Boundary between radiative and convective
        T_r = T_rad
        p_r = p_rad
        convective_switch = 1

      T_old = T_rad * 1.01
      T_new = T_rad


      while((np.absolute((T_new - T_old)/T_new) > 1e-4)):
        p_conv = p0 * np.exp(- z / H(T_new, g))
        T_conv = T_r * pow((p_conv / p_r), Delta)
        T_old = T_new
        T_new = T_old + 0.1 * (T_conv - T_old)

      kr = get_kappa(p_conv, T_conv, fit_kappa)                     # cm^2 / g (pressure from bars to Pa)
      optical_depth = optical_depth + (kr * (p_conv - p_prev)*1e6 / g)[0]
      T_rad = T_conv

      T_prev = T_conv
      p_prev = p_conv

      grad_opt = (kr * (p_conv - p_prev)*1e6 / g)[0] / delta_z

    if (grad_opt > 1e-8):
      delta_z = z_max/100000
    else:
      delta_z = z_max/1000

  return [T_prev, density, H_max, z_max]

M_J = 1.898e30         #Jupiter mass in g
data = np.loadtxt('CO2_data')
X = np.log10(data[:,0])
Y = data[:,1]
Z = np.log10(data[:,2])
Xrbs = X[::41]
Yrbs = Y[0:41]
Zrbs = Z.reshape((41,41))
fit_kappa = interpolate.RectBivariateSpline(Xrbs, Yrbs, Zrbs)

diff_eqs = np.load('diff_eqs.npy', allow_pickle=True)
masses_and_ids = np.load('masses.npy')
ids = masses_and_ids[0]
masses = masses_and_ids[1]
p0 = 100

start_index = int(sys.argv[1])

for i in range(start_index, start_index + 453):
  id_moon = ids[i]
  m_moon = masses[i]
  for j in range(0,1000,10):
    a_moon = diff_eqs[i].y[0,j]
    e_moon = diff_eqs[i].y[1,j]
    time = diff_eqs[i].t[j] / (60*60*24*365*1e6)
    T_surface, rho_max, H_max, z_max = T_profile(a_moon, e_moon, m_moon, p0, fit_kappa)
    with open('output.out','a') as f:
      print(id_moon, time, T_surface, H_max, z_max, rho_max, file=f)
