#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: hexagon, house
#set page(width: auto, height: auto, margin: 5mm, fill: white)
//#set text(font: "Geist")
#set text(font: "Avenir")

#{
  let blob(pos, label, tint: white, lighten:0%, ..args) = node(
    pos,
    align(center, label),
    width: 32mm,
    fill: tint.lighten(60%).lighten(lighten),
    stroke: 1pt + tint.darken(20%).lighten(lighten),
    corner-radius: 5pt,
    ..args,
  )

  // Define colors
  let cellc = rgb("#90caf9")
  let ctxtc = rgb("#ffb74d")

  let norm(pos, tint) = blob(pos, [LayerNorm], tint: tint, lighten:40%)
  let add(pos, tint, ..args) = node(
    pos,
    align(center, [+]),
    width: 6mm,
    fill: tint.lighten(60%).lighten(40%),
    stroke: 1pt + tint.darken(20%).lighten(40%),
    shape: circle,
    ..args,
  )

  diagram(
    spacing: 2mm,
    cell-size: (10mm, 6mm),
    edge-stroke: 0.5mm,
    edge-corner-radius: 3mm,
    mark-scale: 70%,
    axes: (ltr, ttb),

    {
      for (name, x, key, c) in (([Context], -1, "ctxt", ctxtc), ([Cells], 1, "cells", cellc)) {
        let y = 0
        let outd = (ctxt: "l", cells: "r").at(key)
        let outs = (ctxt: -1, cells: 1).at(key)
        let ind = (ctxt: "r", cells: "l").at(key)
        let qkvkw = (
          label-pos: 100% - 1em,
          label-side: left,
          label-size: 3mm,
        )

        blob((x, y), name + [ Output], tint: c, shape: house.with(angle: 10deg))

        edge("<|-")
        y = y + 2

        add((x, y), c)
        edge(outd,"ddd" ,ind, "d", "<|-")
        edge()
        blob((x, y + 1), [SwiGLU], tint: c)
        edge()
        norm((x, y + 2), c)

        edge("<|-")
        y = y + 4

        add((x, y), c)
        edge(outd , "ddddd" , ind , "d", "<|-")
        edge()
        blob((x, y + 1), [Attention], tint: c)
        norm((x, y + 4), c)
        edge(auto, (rel: (0, -0.8)), (rel: (0.4*outs, 0)), "uu", "-|>", [Q], ..qkvkw)
        edge(auto, (rel: (0, -0.8)), (-x+0.4*outs,y+2.5),(rel: (0, -1.5)), "-|>", [V], ..qkvkw)
        edge(auto, (rel: (0, -0.8)), (-x+0.4*outs,y+2.5),(rel:(-0.4*outs,-0.2)),(rel: (0, -1.5)), "-|>", [K], ..qkvkw)

        edge("<|-")
        y = y + 6

        add((x, y), c, name: <SA>)
        edge(outd , "dddd" , ind , "d", "<|-")
        edge()
        blob((x, y + 1), (ctxt: [], cells: [Axial]).at(key) + [Attention], tint: c)
        norm((x, y + 3), c)
        edge(auto, (rel: (0, -0.8)), (rel: (-0.4, 0)), (rel: (0,-1)), "-|>", [Q], ..qkvkw)
        edge("uu","-|>", [K], ..qkvkw)
        edge(auto, (rel: (0, -0.8)), (rel: (0.4, 0)), (rel: (0,-1)), "-|>", [V], ..qkvkw)

        edge("<|-")
        y = y + 6

        blob((x, y), name + [ Input], tint: c, shape: house.with(angle: 10deg))
      }
    },
  )
}
