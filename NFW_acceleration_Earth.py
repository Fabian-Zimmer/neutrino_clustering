import numpy as np
import matplotlib.pyplot as plt
from Shared.shared import Params

class NFWHalo:
    """
    Navarro-Frenk-White (NFW) dark matter halo profile calculator
    """
    
    def __init__(self, M200=1.5e12, c200=12.0, h=0.7):
        """
        Initialize NFW halo parameters
        
        Parameters:
        -----------
        M200 : float
            Virial mass in solar masses (mass within r200)
        c200 : float
            Concentration parameter (r200/rs)
        h : float
            Hubble parameter (H0 = 100h km/s/Mpc)
        """
        self.M200 = M200  # solar masses
        self.c200 = c200
        self.h = h
        
        # Physical constants
        self.G = 6.67430e-11  # m³/kg/s²
        self.M_sun = 1.989e30  # kg
        self.kpc_to_m = 3.086e19  # meters per kpc
        
        # Calculate derived parameters
        self.rho_crit = self._critical_density()
        self.r200 = self._virial_radius()
        self.rs = self.r200 / self.c200
        self.rho0 = self._characteristic_density()
        
    def _critical_density(self):
        """Calculate critical density of the universe"""
        # Critical density in kg/m³
        return 1.878e-26 * self.h**2
    
    def _virial_radius(self):
        """Calculate virial radius r200 in kpc"""
        # r200 = (3*M200 / (4*pi*200*rho_crit))^(1/3)
        return ((3 * self.M200 * self.M_sun) / 
                (4 * np.pi * 200 * self.rho_crit))**(1/3) / self.kpc_to_m
    
    def _characteristic_density(self):
        """Calculate characteristic density rho0"""
        # From NFW normalization
        delta_c = (200/3) * self.c200**3 / (np.log(1 + self.c200) - self.c200/(1 + self.c200))
        return delta_c * self.rho_crit
    
    def density(self, r):
        """
        NFW density profile
        
        Parameters:
        -----------
        r : float or array
            Radius in kpc
            
        Returns:
        --------
        rho : float or array
            Density in kg/m³
        """
        x = r / self.rs
        return self.rho0 / (x * (1 + x)**2)
    
    def enclosed_mass(self, r):
        """
        Mass enclosed within radius r
        
        Parameters:
        -----------
        r : float or array
            Radius in kpc
            
        Returns:
        --------
        M : float or array
            Enclosed mass in kg
        """
        x = r / self.rs
        # NFW enclosed mass formula
        M_nfw = 4 * np.pi * self.rho0 * self.rs**3 * (np.log(1 + x) - x/(1 + x))
        return M_nfw * (self.kpc_to_m)**3
    
    def acceleration(self, r):
        """
        Gravitational acceleration at radius r
        
        Parameters:
        -----------
        r : float or array
            Radius in kpc
            
        Returns:
        --------
        a : float or array
            Acceleration in m/s²
        """
        M_enc = self.enclosed_mass(r)
        r_m = r * self.kpc_to_m  # Convert to meters
        return self.G * M_enc / r_m**2
    
    def acceleration_mm_per_s2(self, r):
        """
        Gravitational acceleration in mm/s²
        
        Parameters:
        -----------
        r : float or array
            Radius in kpc
            
        Returns:
        --------
        a : float or array
            Acceleration in mm/s²
        """
        return self.acceleration(r) * 1000  # Convert m/s² to mm/s²

def main():
    """
    Calculate acceleration at Earth's radius due to Milky Way dark matter halo
    """
    
    # Initialize Milky Way NFW halo with typical parameters
    print("Milky Way Dark Matter Halo - NFW Profile")
    print("=" * 50)
    
    # Milky Way parameters (reasonable estimates)
    M200 = Params.Mvir_MW/Params.Msun      # Solar masses
    c200 = Params.Rvir_MW/Params.Rs_MW        # Concentration parameter
    h = Params.h            # Hubble parameter
    
    halo = NFWHalo(M200=M200, c200=c200, h=h)
    
    print(f"Halo Parameters:")
    print(f"  Virial mass (M200): {M200:.1e} M☉")
    print(f"  Concentration (c200): {c200}")
    print(f"  Virial radius (r200): {halo.r200:.1f} kpc")
    print(f"  Scale radius (rs): {halo.rs:.1f} kpc")
    print(f"  Characteristic density (ρ0): {halo.rho0:.2e} kg/m³")
    print()
    
    # Earth's distance from Galactic center
    r_earth = 8.178  # kpc
    
    # Calculate acceleration at Earth's radius
    accel_ms2 = halo.acceleration(r_earth)
    accel_mm_s2 = halo.acceleration_mm_per_s2(r_earth)
    
    print(f"Results at Earth's Galactocentric Distance:")
    print(f"  Distance from Galactic center: {r_earth} kpc")
    print(f"  Gravitational acceleration: {accel_ms2:.2e} m/s²")
    print(f"  Gravitational acceleration: {accel_mm_s2:.2e} mm/s²")
    print()
    
    # For comparison, calculate total enclosed mass
    M_enc = halo.enclosed_mass(r_earth)
    M_enc_solar = M_enc / halo.M_sun
    
    print(f"Additional Information:")
    print(f"  Enclosed dark matter mass: {M_enc_solar:.2e} M☉")
    print(f"  Fraction of total halo mass: {M_enc_solar/M200:.2f}")
    print()
    
    # Plot the acceleration profile
    r_range = np.logspace(-1, 2, 100)  # 0.1 to 100 kpc
    accel_profile = halo.acceleration_mm_per_s2(r_range)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(r_range, accel_profile, 'b-', linewidth=2, label='NFW Dark Matter Halo')
    plt.axvline(r_earth, color='red', linestyle='--', alpha=0.7, label=f'Earth ({r_earth} kpc)')
    plt.axhline(accel_mm_s2, color='red', linestyle=':', alpha=0.7)
    
    plt.xlabel('Galactocentric Distance (kpc)')
    plt.ylabel('Gravitational Acceleration (mm/s²)')
    plt.title('Milky Way Dark Matter Halo - Gravitational Acceleration Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotation
    plt.annotate(f'a = {accel_mm_s2:.2e} mm/s²', 
                xy=(r_earth, accel_mm_s2), 
                xytext=(r_earth*2, accel_mm_s2*2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()
    
    return accel_mm_s2

if __name__ == "__main__":
    acceleration = main()
    print(f"\nFinal Result: {acceleration:.3e} mm/s²")
