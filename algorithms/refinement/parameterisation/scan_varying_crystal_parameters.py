#
#  Copyright (C) (2013) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.
#

from __future__ import division
from dials.algorithms.refinement.parameterisation.scan_varying_model_parameters \
        import ScanVaryingParameterSet, \
               ScanVaryingModelParameterisation, \
               GaussianSmoother
from scitbx import matrix
from dials.algorithms.refinement import dR_from_axis_and_angle

class ScanVaryingCrystalOrientationParameterisation(ScanVaryingModelParameterisation):
    '''A work-in-progress time-dependent parameterisation for crystal
    orientation, with angles expressed in mrad'''

    def __init__(self, crystal, t_range, num_intervals):

        # The state of a scan varying crystal orientation parameterisation
        # is an orientation
        # matrix '[U](t)', expressed as a function of 'time' t (which could
        # actually be measured by image number in a sequential scan)

        # The initial state is a snapshot of the crystal orientation
        # at the time of initialisation '[U0]', which independent of time.
        # Future states are composed by
        # rotations around axes of the phi-axis frame by Tait-Bryan angles.
        #
        # [U](t) = [Phi3](t)[Phi2](t)[Phi1](t)[U0]

        # Set up the smoother
        smoother = GaussianSmoother(t_range, num_intervals)
        nv = smoother.num_values()

        ### Set up the initial state
        istate = crystal.get_U()

        ### Set up the parameters
        phi1 = ScanVaryingParameterSet(0.0, nv,
                            matrix.col((1., 0., 0.)), 'angle', 'Phi1')
        phi2 = ScanVaryingParameterSet(0.0, nv,
                            matrix.col((0., 1., 0.)), 'angle', 'Phi2')
        phi3 = ScanVaryingParameterSet(0.0, nv,
                            matrix.col((1., 0., 0.)), 'angle', 'Phi3')

        # build the list of parameter sets in a specific, maintained order
        p_list = [phi1, phi2, phi3]

        # set up the list of model objects being parameterised (here
        # just a single crystal model)
        models = [crystal]

        # set up the base class
        ScanVaryingModelParameterisation.__init__(self, models, istate,
                                                  p_list, smoother)

        return

    def get_ds_dp(self, t, only_free = True):
        '''calculate derivatives for model at time t'''

        # Extract orientation from the initial state
        U0 = self._initial_state

        # extract parameter sets from the internal list
        phi1_set, phi2_set, phi3_set = self._param_sets

        # extract angles and other data at time t using the smoother
        phi1, phi1_weights, phi1_sumweights = self._smoother.value_weight(t, phi1_set)
        phi2, phi2_weights, phi2_sumweights = self._smoother.value_weight(t, phi2_set)
        phi3, phi3_weights, phi3_sumweights = self._smoother.value_weight(t, phi3_set)

        # calculate derivatives of angles wrt underlying parameters.
        # FIXME write up notes in orange notebook
        dphi1_dp = [e / phi1_sumweights for e in phi1_weights]
        dphi2_dp = [e / phi2_sumweights for e in phi2_weights]
        dphi3_dp = [e / phi3_sumweights for e in phi3_weights]

        # convert angles to radians
        phi1rad, phi2rad, phi3rad = (phi1 / 1000., phi2 / 1000.,
                                     phi3 / 1000.)

        # compose rotation matrices and their first order derivatives wrt angle
        Phi1 = (phi1_set.axis).axis_and_angle_as_r3_rotation_matrix(phi1rad, deg=False)
        dPhi1_dphi1 = dR_from_axis_and_angle(phi1_set.axis, phi1rad, deg=False)

        Phi2 = (phi2_set.axis).axis_and_angle_as_r3_rotation_matrix(phi2rad, deg=False)
        dPhi2_dphi2 = dR_from_axis_and_angle(phi2_set.axis, phi2rad, deg=False)

        Phi3 = (phi3_set.axis).axis_and_angle_as_r3_rotation_matrix(phi3rad, deg=False)
        dPhi3_dphi3 = dR_from_axis_and_angle(phi3_set.axis, phi3rad, deg=False)

        Phi21 = Phi2 * Phi1
        Phi321 = Phi3 * Phi21

        ### Compose new state

        #newU = Phi321 * U0
        #self._models[0].set_U(newU)

        ### calculate derivatives of the state wrt angle, convert back to mrad
        dU_dphi1 = Phi3 * Phi2 * dPhi1_dphi1 * U0 / 1000.
        dU_dphi2 = Phi3 * dPhi2_dphi2 * Phi1 * U0 / 1000.
        dU_dphi3 = dPhi3_dphi3 * Phi21 * U0 / 1000.

        # calculate derivatives of state wrt underlying parameters
        dU_dp1 = [dU_dphi1 * e for e in dphi1_dp]
        dU_dp2 = [dU_dphi2 * e for e in dphi2_dp]
        dU_dp3 = [dU_dphi3 * e for e in dphi3_dp]

        # return concatenated list of derivatives
        return dU_dp1 + dU_dp2 + dU_dp3

    def get_state(self, t):

        '''Return crystal orientation matrix [U] at time t'''

        # Extract orientation from the initial state
        U0 = self._initial_state

        # extract parameter sets from the internal list
        phi1_set, phi2_set, phi3_set = self._param_sets

        # extract angles and other data at time t using the smoother
        phi1, phi1_weights, phi1_sumweights = self._smoother.value_weight(t, phi1_set)
        phi2, phi2_weights, phi2_sumweights = self._smoother.value_weight(t, phi2_set)
        phi3, phi3_weights, phi3_sumweights = self._smoother.value_weight(t, phi3_set)

        # convert angles to radians
        phi1rad, phi2rad, phi3rad = (phi1 / 1000., phi2 / 1000.,
                                     phi3 / 1000.)

        # compose rotation matrices
        Phi1 = (phi1_set.axis).axis_and_angle_as_r3_rotation_matrix(phi1rad, deg=False)
        Phi2 = (phi2_set.axis).axis_and_angle_as_r3_rotation_matrix(phi2rad, deg=False)
        Phi3 = (phi3_set.axis).axis_and_angle_as_r3_rotation_matrix(phi3rad, deg=False)

        Phi21 = Phi2 * Phi1
        Phi321 = Phi3 * Phi21

        ### Compose new state

        newU = Phi321 * U0
        #self._models[0].set_U(newU)
        # get U(t)
        return newU
