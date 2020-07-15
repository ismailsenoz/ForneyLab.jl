@structuredVariationalRule(:node_type      => GCV{GaussHermite},
                           :outbound_type  => Message{GaussianMeanVariance},
                           :inbound_types  => (Nothing, Message{Gaussian},ProbabilityDistribution),
                           :name           => SVBGCVGaussHermiteOutNGD)

@structuredVariationalRule(:node_type      => GCV{GaussHermite},
                           :outbound_type  => Message{GaussianMeanVariance},
                           :inbound_types  => (Message{Gaussian},Nothing,ProbabilityDistribution),
                           :name           => SVBGCVGaussHermiteMGND)

@structuredVariationalRule(:node_type      => GCV{GaussHermite},
                           :outbound_type  => Message{Function},
                           :inbound_types  => (ProbabilityDistribution,Nothing),
                           :name           => SVBGCVGaussHermiteZDN)

@marginalRule(:node_type      => GCV{GaussHermite},
              :inbound_types  => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
              :name           => MGCVGaussHermiteMGGD)

@structuredVariationalRule(:node_type    => GCV{SphericalRadial},
                         :outbound_type  => Message{GaussianMeanVariance},
                         :inbound_types  => (Nothing, Message{Gaussian},ProbabilityDistribution),
                         :name           => SVBGCVSphericalRadialOutNGD)

@structuredVariationalRule(:node_type    => GCV{SphericalRadial},
                         :outbound_type  => Message{GaussianMeanVariance},
                         :inbound_types  => (Message{Gaussian},Nothing,ProbabilityDistribution),
                         :name           => SVBGCVSphericalRadialMGND)

@structuredVariationalRule(:node_type    => GCV{SphericalRadial},
                         :outbound_type  => Message{Function},
                         :inbound_types  => (ProbabilityDistribution,Nothing),
                         :name           => SVBGCVSphericalRadialZDN)

@marginalRule(:node_type    => GCV{SphericalRadial},
            :inbound_types  => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
            :name           => MGCVSphericalRadialMGGD)

@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Nothing, Message{Gaussian},ProbabilityDistribution),
                          :name           => SVBGCVLaplaceOutNGD)

@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Message{Gaussian},Nothing,ProbabilityDistribution),
                          :name           => SVBGCVLaplaceMGND)

@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (ProbabilityDistribution,Nothing),
                          :name           => SVBGCVLaplaceZDN)

@marginalRule(:node_type      => GCV{Laplace},
             :inbound_types  => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
             :name           => MGCVLaplaceMGGD)
