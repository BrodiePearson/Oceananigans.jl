using JLD2, Plots
using Statistics

l_fields_file = jldopen("langmuir_turbulence_fields.jld2")
s_fields_file = jldopen("shear_turbulence_fields.jld2")

l_data_av=jldopen("langmuir_turbulence_averages.jld2")
s_data_av=jldopen("shear_turbulence_averages.jld2")
l_iterations = parse.(Int, keys(l_data_av["timeseries/t"]))
s_iterations = parse.(Int, keys(s_data_av["timeseries/t"]))

l_end_time = last(l_iterations)
s_end_time = last(s_iterations)
#l_end_time = l_iterations[end-10]
#s_end_time = s_iterations[end-10]

l_T_init = l_fields_file["timeseries/T/0"]
l_S_init = l_fields_file["timeseries/S/0"]
s_T_init = s_fields_file["timeseries/T/0"]
s_S_init = s_fields_file["timeseries/S/0"]

l_Tₑ = l_fields_file["timeseries/T/$l_end_time"]
l_Sₑ = l_fields_file["timeseries/S/$l_end_time"]
s_Tₑ = s_fields_file["timeseries/T/$s_end_time"]
s_Sₑ = s_fields_file["timeseries/S/$s_end_time"]

l_mean_Tₑ = mean(mean(l_Tₑ, dims=1), dims=2)
l_mean_T_init = mean(mean(l_T_init, dims=1), dims=2)
l_mean_Sₑ = mean(mean(l_Sₑ, dims=1), dims=2)
l_mean_S_init = mean(mean(l_S_init, dims=1), dims=2)

s_mean_Tₑ = mean(mean(s_Tₑ, dims=1), dims=2)
s_mean_T_init = mean(mean(s_T_init, dims=1), dims=2)
s_mean_Sₑ = mean(mean(s_Sₑ, dims=1), dims=2)
s_mean_S_init = mean(mean(s_S_init, dims=1), dims=2)

depths = -l_fields_file["grid/Lz"] .+range(0,stop=l_fields_file["grid/Lz"],length=l_fields_file["grid/Nz"])
plot_T_evolution = plot([l_mean_T_init[1,1,:], l_mean_Tₑ[1,1,:], s_mean_Tₑ[1,1,:] ], depths;
                        xlims = (-1.42, -1.38),
                        ylims = (-20, 0),
                         xlabel = "Pot. Temp. (C)",
                         ylabel = "",
                         linestyle = [:solid :solid :solid],
                         linecolor = [:black :blue :red],
                         linewidth = 8,
                         label = :false,
                         #label = ["t = 0 (lang)" "t = 48 hours (lang)" "t = 0 (shear)" "t = 48 hours (shear)"],
                         legend = :bottomleft)

plot_S_evolution = plot([s_mean_S_init[1,1,:], l_mean_Sₑ[1,1,:], s_mean_Sₑ[1,1,:] ], depths;
                         xlims = (29.05, 29.08),
                         ylims = (-20, 0),
                          xlabel = "S (g/kg)",
                          ylabel = "",
                          linestyle = [:solid :solid :solid],
                          linecolor = [:black :blue :red],
                          linewidth = 8,
                          label = :false,
                          #label = ["t = 0 (lang)" "t = 48 hours (lang)" "t = 0 (shear)" "t = 48 hours (shear)"],
                          legend = :bottomleft)



l_wT_snapshot = l_data_av["timeseries/wT/$l_end_time"][1, 1, :]
l_wS_snapshot = l_data_av["timeseries/wS/$l_end_time"][1, 1, :]
l_ww_snapshot = l_data_av["timeseries/ww/$l_end_time"][1, 1, :]

s_wT_snapshot = s_data_av["timeseries/wT/$s_end_time"][1, 1, :]
s_wS_snapshot = s_data_av["timeseries/wS/$s_end_time"][1, 1, :]
s_ww_snapshot = s_data_av["timeseries/ww/$s_end_time"][1, 1, :]

plot_w² = plot([l_ww_snapshot[2:end], s_ww_snapshot[2:end]], depths;
                        xlims = (0, 5e-5),
                        ylims = (-20, 0),
                        xlabel = "w² (m² s⁻²)",
                        ylabel = "",
                        #label = ["Langmuir" "Shear"],
                        linestyle = [:solid :solid],
                        linecolor = [:blue :red],
                        linewidth = 8,
                        label = :none,
                        legend = :bottomright)

tracer_fluxes_plot = plot([l_wT_snapshot[2:end], s_wT_snapshot[2:end]], depths,
                      #label = ["wT Langmuir" "wS Langmuir" "wT Shear" "wS Shear"],
                      #legend = :bottom,
                      ylims = (-20, 0),
                      xlabel = "Tracer fluxes (m² s⁻²)",
                      linestyle = [:solid :solid],
                      linecolor = [:blue :red],
                      linewidth = 8,
                      label = :none,
                      ylabel = "")

l = @layout [a b c d]
 plot_changes = plot(plot_T_evolution, plot_w², tracer_fluxes_plot, plot_S_evolution, layout=l, size=(3200, 800),
      title = ["T(z, t=end) (m/s)" "w²(z, t=end) (m/s)" "w'T'(z, t=end) (m/s)" "S(z, t=end) (m/s)" ])

savefig("examples/proposal_subsurfaceheat_figure.png")
